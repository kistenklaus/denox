include_guard(GLOBAL)

if(DENOX_STATIC_SQLITE3)
    include(FetchContent)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        list(
            APPEND
            DENOX_SQLITE_FETCHCONTENT_ARGS
                DOWNLOAD_EXTRACT_TIMESTAMP
                    TRUE
        )
    endif()

    FetchContent_Declare(
        sqlite3_amalgamation
        URL "https://www.sqlite.org/2026/sqlite-amalgamation-3530100.zip"
        URL_HASH "SHA3_256=3c07136e4f6b5dd0c395be86455014039597bc65b6851f7111e88f71b6e06114"
        GIT_PROGRESS TRUE
        ${DENOX_SQLITE_FETCHCONTENT_ARGS}
    )

    FetchContent_MakeAvailable(sqlite3_amalgamation)

    file(
        GLOB_RECURSE
        DENOX_SQLITE3_C
        CONFIGURE_DEPENDS
        "${sqlite3_amalgamation_SOURCE_DIR}/sqlite3.c"
        "${sqlite3_amalgamation_SOURCE_DIR}/*/sqlite3.c"
    )

    list(
        LENGTH
        DENOX_SQLITE3_C
        DENOX_SQLITE3_C_COUNT
    )

    if(NOT DENOX_SQLITE3_C_COUNT EQUAL 1)
        message(
            FATAL_ERROR
            "Expected exactly one sqlite3.c in ${sqlite3_amalgamation_SOURCE_DIR}, "
            "found ${DENOX_SQLITE3_C_COUNT}: ${DENOX_SQLITE3_C}"
        )
    endif()

    list(
        GET
        DENOX_SQLITE3_C
        0
        DENOX_SQLITE3_C
    )

    get_filename_component(
        DENOX_SQLITE_SOURCE_DIR
        "${DENOX_SQLITE3_C}"
        DIRECTORY
    )

    add_library(denox_sqlite3 STATIC
        "${DENOX_SQLITE3_C}"
    )

    add_library(SQLite::SQLite3 ALIAS denox_sqlite3)

    target_include_directories(denox_sqlite3
        PUBLIC
            "${DENOX_SQLITE_SOURCE_DIR}"
    )

    target_compile_definitions(denox_sqlite3
        PRIVATE
            SQLITE_THREADSAFE=1
            SQLITE_OMIT_LOAD_EXTENSION
    )

    set_target_properties(denox_sqlite3
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
else()
    find_package(SQLite3 REQUIRED)
endif()

if(NOT TARGET SQLite::SQLite3)
  message(FATAL_ERROR "SQLite::SQLite3 imported target missing")
endif()
