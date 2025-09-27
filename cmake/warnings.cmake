
function(denox_enable_strict_warnings tgt)
  if (MSVC)
    target_compile_options(${tgt} PRIVATE
      /W4               # high warning level
      /permissive-      # strict standard conformance
      /Zc:__cplusplus   # correct __cplusplus macro
      /Zc:preprocessor  # modern preprocessor
      /w14242 /w14254 /w14263 /w14265 /w14287
      /w14296 /w14311 /w14826 /w44062 /w44242
      /w14545 /w14546 /w14547 /w14549 /w14905
      /w14906 /w14928   # promote some noisy ones to level 4-as-errors
      $<$<BOOL:${DENOX_WARN_AS_ERR}>:/WX>
      /external:W0      # silence external headers
      /external:anglebrackets
    )
    # Optional: common MSVC noise reduction
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${tgt} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wconversion -Wsign-conversion
      -Wold-style-cast -Woverloaded-virtual
      -Wnon-virtual-dtor
      -Wcast-qual -Wcast-align
      -Wformat=2 -Wno-null-dereference -Wdouble-promotion
      -Wimplicit-fallthrough -Wmissing-field-initializers
      -Wswitch-enum -Wundef
      -Wrange-loop-construct -Wextra-semi -Wnewline-eof
      $<$<BOOL:${DENOX_WARN_AS_ERR}>:-Werror>
    )
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wconversion -Wsign-conversion
      -Wold-style-cast -Woverloaded-virtual
      -Wnon-virtual-dtor
      -Wcast-qual -Wcast-align=strict
      -Wformat=2 -Wno-null-dereference -Wdouble-promotion
      -Wimplicit-fallthrough -Wmissing-field-initializers
      -Wswitch-enum -Wundef
      -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast
      $<$<BOOL:${DENOX_WARN_AS_ERR}>:-Werror>
    )
  endif()
endfunction()
