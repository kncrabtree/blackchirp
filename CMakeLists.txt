cmake_minimum_required(VERSION 3.5)

project(blackchirp-tests LANGUAGES CXX)


find_package(Qt5Test REQUIRED)
find_package(Qt5Gui REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(tst_settingsstoragetest tests/tst_settingsstoragetest.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_settingsstoragetest COMMAND tst_settingsstoragetest)

add_executable(tst_headerstoragetest tests/tst_headerstoragetest.cpp src/data/storage/headerstorage.cpp)
add_test(NAME tst_headerstoragetest COMMAND tst_headerstoragetest)

add_executable(tst_blackchirpcsvtest tests/tst_blackchirpcsv.cpp src/data/storage/blackchirpcsv.cpp)
add_test(NAME tst_blackchirpcsvtest COMMAND tst_blackchirpcsvtest)

target_link_libraries(tst_settingsstoragetest PRIVATE Qt5::Gui Qt5::Test)
target_link_libraries(tst_headerstoragetest PRIVATE Qt5::Gui Qt5::Test)
target_link_libraries(tst_blackchirpcsvtest PRIVATE Qt5::Gui Qt5::Test)
