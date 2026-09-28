# Custom ESPHome Branch Guidelines

1. **Custom `AddressableLight` Methods (Dithering & Raw)**: This repository is a custom fork of ESPHome specifically modified to include spatial dithering and raw hardware buffer writing methods (like `set_dithered_color`). 
2. **Backward Compatibility via SFINAE**: External configuration repos (like `esphome-configs`) often share custom C++ headers between devices running this custom fork and devices running standard ESPHome. When adjusting custom C++ APIs in this fork, maintain method signatures that can be cleanly resolved or bypassed using C++ SFINAE template metaprogramming from the user space. This ensures custom capabilities are strictly opt-in and won't hard-crash the C++ compiler for users falling back to the main branch.
