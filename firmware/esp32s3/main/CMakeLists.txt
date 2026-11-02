
idf_component_register(
    SRCS
        "main.c"
        "feature.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp_driver_gpio
        esp_driver_i2s
        esp_timer
        esp_ringbuf
        espressif__esp-dsp
)
