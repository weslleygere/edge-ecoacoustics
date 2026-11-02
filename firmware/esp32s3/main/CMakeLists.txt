
idf_component_register(
    SRCS "main.c" "feature.c" "inference.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_i2s esp_timer esp_ringbuf esp_psram
             espressif__esp-dsp espressif__esp-tflite-micro
)

target_add_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_LIST_DIR}/../model/cnn_logmel_float32.tflite" BINARY ALIGN 16)
target_add_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_LIST_DIR}/../model/normalization.bin" BINARY ALIGN 16)
