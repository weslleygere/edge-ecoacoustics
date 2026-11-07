/*
 * Edge Ecoacoustics - Wi-Fi + MQTT/TLS
 * ESP32-S3 / ESP-IDF 6.x
 *
 * Diagnostics for Wi-Fi, IP, NTP, MQTT and telemetry delivery.
 * Maintains the existing topic and JSON payload format.
 */

#include "network.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_ip_addr.h"
#include "esp_netif_sntp.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "mqtt_client.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#define MQTT_URI "mqtts://x709811d.ala.eu-central-1.emqxsl.com:8883"
#define MQTT_TOPIC "eco/esp32s3_01/indices"
#define WIFI_READY BIT0
#define MQTT_READY BIT1
#define QUEUE_LENGTH 32
#define CONNECTION_LOG_INTERVAL_MS 15000

static const char *TAG = "eco_network";
static QueueHandle_t telemetry_queue = NULL;
static EventGroupHandle_t network_events = NULL;
static esp_mqtt_client_handle_t mqtt_client = NULL;
static uint32_t dropped = 0;

extern const uint8_t emqx_ca_start[] asm("_binary_emqxsl_ca_crt_start");

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    (void)arg;

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Wi-Fi started | SSID=%s", CONFIG_ECO_WIFI_SSID);
        esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Wi-Fi connect failed: %s", esp_err_to_name(err));
        }
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_CONNECTED) {
        ESP_LOGI(TAG, "Connected to access point; waiting for IP");
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        const wifi_event_sta_disconnected_t *event =
            (const wifi_event_sta_disconnected_t *)data;

        xEventGroupClearBits(network_events, WIFI_READY | MQTT_READY);
        ESP_LOGW(TAG, "Wi-Fi disconnected | reason=%u; reconnecting",
                 event != NULL ? (unsigned)event->reason : 0U);

        esp_err_t err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Wi-Fi reconnect failed: %s", esp_err_to_name(err));
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *event = (const ip_event_got_ip_t *)data;
        if (event != NULL) {
            ESP_LOGI(TAG, "Wi-Fi connected | IP=" IPSTR,
                     IP2STR(&event->ip_info.ip));
        } else {
            ESP_LOGI(TAG, "Wi-Fi connected | IP obtained");
        }
        xEventGroupSetBits(network_events, WIFI_READY);
    }
}

static void mqtt_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    (void)arg;
    (void)base;

    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)data;

    if (id == MQTT_EVENT_CONNECTED) {
        xEventGroupSetBits(network_events, MQTT_READY);
        ESP_LOGI(TAG, "MQTT connected | topic=%s", MQTT_TOPIC);
    } else if (id == MQTT_EVENT_DISCONNECTED) {
        xEventGroupClearBits(network_events, MQTT_READY);
        ESP_LOGW(TAG, "MQTT disconnected");
    } else if (id == MQTT_EVENT_PUBLISHED) {
        ESP_LOGI(TAG, "MQTT acknowledged | id=%d", event->msg_id);
    } else if (id == MQTT_EVENT_ERROR) {
        ESP_LOGW(TAG, "MQTT connection error");
    }
}

static void network_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "Network task started");

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    assert(esp_netif_create_default_wifi_sta() != NULL);

    wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
    wifi_config_t wifi_cfg = {
        .sta = {
            .ssid = CONFIG_ECO_WIFI_SSID,
            .password = CONFIG_ECO_WIFI_PASSWORD,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL));

    esp_sntp_config_t sntp_cfg = ESP_NETIF_SNTP_DEFAULT_CONFIG("pool.ntp.org");
    sntp_cfg.start = false;
    ESP_ERROR_CHECK(esp_netif_sntp_init(&sntp_cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_LOGI(TAG, "Starting Wi-Fi");
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Waiting for Wi-Fi connection and IP");
    while ((xEventGroupWaitBits(network_events, WIFI_READY,
                                pdFALSE, pdTRUE,
                                pdMS_TO_TICKS(CONNECTION_LOG_INTERVAL_MS))
            & WIFI_READY) == 0) {
        ESP_LOGW(TAG, "Still waiting for Wi-Fi/IP");
    }

    ESP_LOGI(TAG, "Wi-Fi ready; starting NTP synchronization");
    ESP_ERROR_CHECK(esp_netif_sntp_start());

    while (esp_netif_sntp_sync_wait(pdMS_TO_TICKS(30000)) != ESP_OK) {
        ESP_LOGW(TAG, "Waiting for NTP synchronization");
    }
    ESP_LOGI(TAG, "System clock synchronized");

    const esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_URI,
        .broker.verification.certificate = (const char *)emqx_ca_start,
        .credentials.username = "eco",
        .credentials.authentication.password = CONFIG_ECO_MQTT_PASSWORD,
        .session.keepalive = 60,
        .outbox.limit = 16384,
        .task.stack_size = 8192,
    };

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    assert(mqtt_client != NULL);

    ESP_ERROR_CHECK(esp_mqtt_client_register_event(
        mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_LOGI(TAG, "Starting MQTT/TLS connection");
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));

    while (true) {
        if ((xEventGroupWaitBits(network_events, MQTT_READY,
                                 pdFALSE, pdTRUE,
                                 pdMS_TO_TICKS(CONNECTION_LOG_INTERVAL_MS))
             & MQTT_READY) == 0) {
            ESP_LOGW(TAG, "Still waiting for MQTT connection | pending=%u",
                     (unsigned)uxQueueMessagesWaiting(telemetry_queue));
            continue;
        }

        eco_telemetry_t item;
        if (xQueueReceive(telemetry_queue, &item,
                          pdMS_TO_TICKS(1000)) != pdTRUE) {
            continue;
        }

        struct timeval now;
        gettimeofday(&now, NULL);

        int64_t now_ms = (int64_t)now.tv_sec * 1000 + now.tv_usec / 1000;
        int64_t ts_ms = now_ms - (esp_timer_get_time() - item.end_us) / 1000;

        // Single network task; static buffer avoids growing the 6-KiB stack.
        static char payload[1536];
        int len = snprintf(payload, sizeof(payload),
            "{\"device\":\"esp32s3_01\",\"ts_ms\":%" PRId64
            ",\"seq\":%" PRIu32
            ",\"bio\":%.4f,\"geo\":%.4f,\"antro\":%.4f"
            ",\"rms\":%.6f,\"inference_ms\":%.2f"
            ",\"errors\":%" PRIu32 ",\"overflows\":%" PRIu32
            ",\"dropped\":%" PRIu32,
            ts_ms, item.seq, item.bio, item.geo, item.antro,
            item.rms, item.inference_ms, item.errors, item.overflows,
            __atomic_load_n(&dropped, __ATOMIC_RELAXED));

        if (len < 0 || len >= (int)sizeof(payload)) {
            __atomic_fetch_add(&dropped, 1, __ATOMIC_RELAXED);
            ESP_LOGE(TAG, "Telemetry payload too large");
            continue;
        }

        // IDs refer to model/prompts.txt; text labels stay outside firmware.
        static const char *const groups[ECO_PROMPT_GROUPS] = {
            "bio", "geo", "antro"
        };
        size_t used = (size_t)len;
        bool valid = true;
        for (int g = 0; g < ECO_PROMPT_GROUPS && valid; ++g) {
            for (int k = 0; k < ECO_TOP_K; ++k) {
                const eco_prompt_rank_t *p = &item.top[g][k];
                if (p->id >= 41) { valid = false; break; }
                int written = snprintf(payload + used, sizeof(payload) - used,
                    ",\"%s_top%d_id\":%u,\"%s_top%d_cosine\":%.5f"
                    ",\"%s_top%d_score\":%.5f",
                    groups[g], k + 1, (unsigned)p->id,
                    groups[g], k + 1, (double)p->cosine,
                    groups[g], k + 1, (double)p->score);
                if (written < 0 || (size_t)written >= sizeof(payload) - used) {
                    valid = false;
                    break;
                }
                used += (size_t)written;
            }
        }
        if (!valid || used + 2 > sizeof(payload)) {
            __atomic_fetch_add(&dropped, 1, __ATOMIC_RELAXED);
            ESP_LOGE(TAG, "Top prompt JSON too large or invalid ID");
            continue;
        }
        payload[used++] = '}';
        payload[used] = '\0';

        int msg_id = esp_mqtt_client_enqueue(
            mqtt_client, MQTT_TOPIC, payload, (int)used, 1, 0, true);

        if (msg_id < 0) {
            if (xQueueSendToFront(telemetry_queue, &item, 0) != pdTRUE) {
                __atomic_fetch_add(&dropped, 1, __ATOMIC_RELAXED);
            }
            ESP_LOGW(TAG, "MQTT enqueue failed: %d", msg_id);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        ESP_LOGI(TAG, "MQTT enqueued | seq=%" PRIu32 " | id=%d",
                 item.seq, msg_id);
    }
}

esp_err_t eco_network_init(void)
{
    telemetry_queue = xQueueCreate(QUEUE_LENGTH, sizeof(eco_telemetry_t));
    network_events = xEventGroupCreate();

    if (!telemetry_queue || !network_events) {
        ESP_LOGE(TAG, "Cannot allocate network queue/events");
        return ESP_ERR_NO_MEM;
    }

    return xTaskCreate(network_task, "eco_network", 6144,
                       NULL, 3, NULL) == pdPASS
        ? ESP_OK : ESP_ERR_NO_MEM;
}

bool eco_network_publish(const eco_telemetry_t *data)
{
    if (data == NULL || telemetry_queue == NULL) {
        ESP_LOGE(TAG, "Telemetry not ready");
        return false;
    }

    if (xQueueSend(telemetry_queue, data, 0) == pdTRUE) {
        ESP_LOGI(TAG, "Telemetry queued | seq=%" PRIu32 " | pending=%u",
                 data->seq, (unsigned)uxQueueMessagesWaiting(telemetry_queue));
        return true;
    }

    uint32_t total = __atomic_add_fetch(&dropped, 1, __ATOMIC_RELAXED);
    ESP_LOGW(TAG, "Telemetry queue full | dropped=%" PRIu32, total);
    return false;
}
