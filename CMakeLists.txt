cmake_minimum_required(VERSION 3.5)

project(blackchirp-tests LANGUAGES CXX)


find_package(Qt6Test REQUIRED)
find_package(Qt6Gui REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

enable_testing()

add_executable(tst_settingsstoragetest tests/tst_settingsstoragetest.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_settingsstoragetest COMMAND tst_settingsstoragetest)

add_executable(tst_headerstoragetest tests/tst_headerstoragetest.cpp src/data/storage/headerstorage.cpp src/data/storage/settingsstorage.cpp src/data/storage/blackchirpcsv.cpp src/data/experiment/fid.cpp src/data/analysis/analysis.cpp)
add_test(NAME tst_headerstoragetest COMMAND tst_headerstoragetest)

add_executable(tst_blackchirpcsvtest tests/tst_blackchirpcsv.cpp src/data/storage/blackchirpcsv.cpp src/data/storage/settingsstorage.cpp src/data/experiment/fid.cpp src/data/analysis/analysis.cpp)
add_test(NAME tst_blackchirpcsvtest COMMAND tst_blackchirpcsvtest)

add_executable(tst_spcatparser tests/tst_spcatparser.cpp src/data/experiment/spcatparser.cpp src/data/experiment/catalogdata.cpp src/data/experiment/catalogparser.cpp src/data/experiment/catalogparserregistry.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_spcatparser COMMAND tst_spcatparser)

target_link_libraries(tst_settingsstoragetest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_headerstoragetest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_blackchirpcsvtest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_spcatparser PRIVATE Qt6::Gui Qt6::Test)
