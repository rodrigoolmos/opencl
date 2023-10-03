cmake_minimum_required(VERSION 3.0)
project(TuProyecto)

# Especifica la versión del lenguaje C que deseas utilizar
set(CMAKE_C_STANDARD 11)

# Agrega el ejecutable
add_executable(mi_programa main.c)

# Directorio donde se encuentran las bibliotecas DLL de opencl
target_link_libraries(mi_programa 
PUBLIC ${OpenCL_LIBRARY} OpenCL m)

