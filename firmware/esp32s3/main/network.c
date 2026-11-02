
#include "network.h"

#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
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

static const char *TAG = "eco_network";
static QueueHandle_t telemetry_queue;
static EventGroupHandle_t network_events;
static esp_mqtt_client_handle_t mqtt_client;
static volatile uint32_t dropped = 0;

extern const uint8_t emqx_ca_start[] asm("_binary_emqxsl_ca_crt_start");

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    (void)data;

    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(network_events, WIFI_READY | MQTT_READY);
        ESP_LOGW(TAG, "Wi-Fi disconnected; reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(network_events, WIFI_READY);
        ESP_LOGI(TAG, "Wi-Fi connected");
    }
}

static void mqtt_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    (void)base;
    (void)data;

    if (id == MQTT_EVENT_CONNECTED) {
        xEventGroupSetBits(network_events, MQTT_READY);
        ESP_LOGI(TAG, "MQTT connected");
    } else if (id == MQTT_EVENT_DISCONNECTED) {
        xEventGroupClearBits(network_events, MQTT_READY);
        ESP_LOGW(TAG, "MQTT disconnected");
    } else if (id == MQTT_EVENT_ERROR) {
        ESP_LOGW(TAG, "MQTT connection error");
    }
}

static void network_task(void *arg)
{
    (void)arg;

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
    ESP_ERROR_CHECK(esp_wifi_start());

    xEventGroupWaitBits(network_events, WIFI_READY, pdFALSE, pdTRUE, portMAX_DELAY);
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
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));

    while (true) {
        xEventGroupWaitBits(network_events, MQTT_READY, pdFALSE, pdTRUE, portMAX_DELAY);

        eco_telemetry_t item;
        if (xQueueReceive(telemetry_queue, &item, pdMS_TO_TICKS(1000)) != pdTRUE) {
            continue;
        }

        struct timeval now;
        gettimeofday(&now, NULL);

        int64_t now_ms = (int64_t)now.tv_sec * 1000 + now.tv_usec / 1000;
        int64_t ts_ms = now_ms - (esp_timer_get_time() - item.end_us) / 1000;

        char payload[256];
        int len = snprintf(payload, sizeof(payload),
            "{\"device\":\"esp32s3_01\",\"ts_ms\":%" PRId64
            ",\"seq\":%" PRIu32
            ",\"bio\":%.4f,\"geo\":%.4f,\"antro\":%.4f"
            ",\"rms\":%.6f,\"inference_ms\":%.2f"
            ",\"errors\":%" PRIu32 ",\"overflows\":%" PRIu32
            ",\"dropped\":%" PRIu32 "}",
            ts_ms, item.seq, item.bio, item.geo, item.antro,
            item.rms, item.inference_ms, item.errors, item.overflows, dropped);

        if (len < 0 || len >= sizeof(payload)) {
            dropped++;
            ESP_LOGE(TAG, "Telemetry payload too large");
            continue;
        }

        int msg_id = esp_mqtt_client_enqueue(
            mqtt_client, MQTT_TOPIC, payload, len, 1, 0, true);

        if (msg_id < 0) {
            if (xQueueSendToFront(telemetry_queue, &item, 0) != pdTRUE) {
                dropped++;
            }
            ESP_LOGW(TAG, "MQTT enqueue failed: %d", msg_id);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        ESP_LOGI(TAG, "MQTT enqueued | seq=%" PRIu32 " | id=%d", item.seq, msg_id);
    }
}

esp_err_t eco_network_init(void)
{
    telemetry_queue = xQueueCreate(QUEUE_LENGTH, sizeof(eco_telemetry_t));
    network_events = xEventGroupCreate();

    if (!telemetry_queue || !network_events) return ESP_ERR_NO_MEM;

    return xTaskCreate(network_task, "eco_network", 6144, NULL, 3, NULL) == pdPASS
        ? ESP_OK : ESP_ERR_NO_MEM;
}

bool eco_network_publish(const eco_telemetry_t *data)
{
    if (xQueueSend(telemetry_queue, data, 0) == pdTRUE) return true;
    dropped++;
    ESP_LOGW(TAG, "Telemetry queue full | dropped=%" PRIu32, dropped);
    return false;
}
