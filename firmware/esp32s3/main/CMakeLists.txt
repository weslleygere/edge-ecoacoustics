idf_component_register(
    SRCS "main.c" "feature.c" "inference.cpp" "network.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_i2s esp_timer esp_psram
             espressif__esp-dsp espressif__esp-tflite-micro
             esp_wifi esp_netif esp_event nvs_flash espressif__mqtt
    EMBED_TXTFILES "emqxsl-ca.crt"
)

target_add_binary_data(${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/../model/tinyclap_phinet_int8.tflite"
    BINARY ALIGN 16)
target_add_binary_data(${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/../model/tinyclap_text_assets.bin"
    BINARY ALIGN 16)
target_add_binary_data(${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/../model/tinyclap_mel_filters.bin"
    BINARY ALIGN 16)
