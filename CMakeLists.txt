cmake_minimum_required(VERSION 3.5)

project(blackchirp-tests LANGUAGES CXX)

# Define version constants (from blackchirp.pro)
set(BC_MAJOR_VERSION 1)
set(BC_MINOR_VERSION 0)
set(BC_PATCH_VERSION 0)
set(BC_RELEASE_VERSION "devel")

# Get git build version
execute_process(
    COMMAND git describe --always
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE BC_BUILD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(Qt6Test REQUIRED)
find_package(Qt6Gui REQUIRED)
find_package(Qt6Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

enable_testing()

# Function to add version definitions to test targets
function(add_version_definitions target_name)
    target_compile_definitions(${target_name} PRIVATE
        BC_MAJOR_VERSION=${BC_MAJOR_VERSION}
        BC_MINOR_VERSION=${BC_MINOR_VERSION}
        BC_PATCH_VERSION=${BC_PATCH_VERSION}
        BC_RELEASE_VERSION=${BC_RELEASE_VERSION}
        BC_BUILD_VERSION="${BC_BUILD_VERSION}"
    )
endfunction()

add_executable(tst_settingsstoragetest tests/tst_settingsstoragetest.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_settingsstoragetest COMMAND tst_settingsstoragetest)

add_executable(tst_headerstoragetest tests/tst_headerstoragetest.cpp src/data/storage/headerstorage.cpp src/data/storage/settingsstorage.cpp src/data/storage/blackchirpcsv.cpp src/data/experiment/fid.cpp src/data/analysis/analysis.cpp)
add_test(NAME tst_headerstoragetest COMMAND tst_headerstoragetest)

add_executable(tst_blackchirpcsvtest tests/tst_blackchirpcsv.cpp src/data/storage/blackchirpcsv.cpp src/data/storage/settingsstorage.cpp src/data/experiment/fid.cpp src/data/analysis/analysis.cpp)
add_test(NAME tst_blackchirpcsvtest COMMAND tst_blackchirpcsvtest)

add_executable(tst_spcatparser tests/tst_spcatparser.cpp src/data/experiment/spcatparser.cpp src/data/experiment/catalogdata.cpp src/data/experiment/catalogparser.cpp src/data/experiment/fileparser.cpp src/data/experiment/catalogparserregistry.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_spcatparser COMMAND tst_spcatparser)

add_executable(tst_xiamparser tests/tst_xiamparser.cpp src/data/experiment/xiamparser.cpp src/data/experiment/catalogdata.cpp src/data/experiment/catalogparser.cpp src/data/experiment/fileparser.cpp src/data/experiment/catalogparserregistry.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_xiamparser COMMAND tst_xiamparser)

add_executable(tst_genericxyparser tests/tst_genericxyparser.cpp src/data/experiment/genericxyparser.cpp src/data/experiment/genericxydata.cpp src/data/experiment/fileparser.cpp src/data/storage/settingsstorage.cpp)
add_test(NAME tst_genericxyparser COMMAND tst_genericxyparser)

# Overlay processing architecture tests
add_executable(tst_overlayprocessing tests/tst_overlayprocessing.cpp 
    src/gui/overlay/overlaytypespecificwidget.cpp
    src/gui/overlay/bcexpoverlaywidget.cpp
    src/gui/overlay/catalogoverlaywidget.cpp
    src/gui/overlay/unifiedoverlaywidget.cpp
    src/gui/overlay/unifiedoverlaydialog.cpp
    src/data/experiment/overlaybase.cpp
    src/data/experiment/overlaytypes.cpp
    src/data/processing/overlayprocessmanager.cpp
    src/data/processing/overlayoperation.cpp
    src/data/analysis/ftworker.cpp
    src/data/analysis/ft.cpp
    src/data/analysis/analysis.cpp
    src/data/experiment/catalogdata.cpp
    src/data/experiment/catalogparser.cpp
    src/data/experiment/fileparser.cpp
    src/data/experiment/catalogparserregistry.cpp
    src/data/experiment/spcatparser.cpp
    src/data/experiment/xiamparser.cpp
    src/data/storage/blackchirpcsv.cpp
    src/data/storage/settingsstorage.cpp
    src/data/storage/overlaystorage.cpp
    src/data/experiment/experiment.cpp
    src/data/experiment/fid.cpp)
add_test(NAME tst_overlayprocessing COMMAND tst_overlayprocessing)

add_executable(tst_unifiedoverlaydialog tests/tst_unifiedoverlaydialog.cpp
    src/gui/overlay/unifiedoverlaydialog.cpp
    src/gui/overlay/unifiedoverlaywidget.cpp
    src/gui/overlay/overlaytypespecificwidget.cpp
    src/gui/overlay/bcexpoverlaywidget.cpp
    src/gui/overlay/catalogoverlaywidget.cpp
    src/data/experiment/overlaybase.cpp
    src/data/experiment/overlaytypes.cpp
    src/data/processing/overlayprocessmanager.cpp
    src/data/processing/overlayoperation.cpp
    src/data/analysis/ft.cpp
    src/data/analysis/analysis.cpp
    src/data/experiment/catalogdata.cpp
    src/data/experiment/catalogparser.cpp
    src/data/experiment/fileparser.cpp
    src/data/experiment/catalogparserregistry.cpp
    src/data/experiment/spcatparser.cpp
    src/data/experiment/xiamparser.cpp
    src/data/storage/blackchirpcsv.cpp
    src/data/storage/overlaystorage.cpp
    src/data/storage/settingsstorage.cpp
    src/data/experiment/fid.cpp)
add_test(NAME tst_unifiedoverlaydialog COMMAND tst_unifiedoverlaydialog)

add_executable(tst_overlayoperations tests/tst_overlayoperations.cpp
    src/gui/overlay/overlaytypespecificwidget.cpp
    src/gui/overlay/bcexpoverlaywidget.cpp
    src/gui/overlay/catalogoverlaywidget.cpp
    src/data/processing/overlayprocessmanager.cpp
    src/data/processing/overlayoperation.cpp
    src/data/experiment/overlaybase.cpp
    src/data/experiment/overlaytypes.cpp
    src/data/analysis/ft.cpp
    src/data/analysis/analysis.cpp
    src/data/experiment/catalogdata.cpp
    src/data/experiment/catalogparser.cpp
    src/data/experiment/fileparser.cpp
    src/data/experiment/catalogparserregistry.cpp
    src/data/experiment/spcatparser.cpp
    src/data/experiment/xiamparser.cpp
    src/data/storage/blackchirpcsv.cpp
    src/data/storage/settingsstorage.cpp
    src/data/experiment/fid.cpp)
add_test(NAME tst_overlayoperations COMMAND tst_overlayoperations)

target_link_libraries(tst_settingsstoragetest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_headerstoragetest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_blackchirpcsvtest PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_spcatparser PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_xiamparser PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_genericxyparser PRIVATE Qt6::Gui Qt6::Test)
target_link_libraries(tst_overlayprocessing PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)
target_link_libraries(tst_unifiedoverlaydialog PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)
target_link_libraries(tst_overlayoperations PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)

# Add version definitions to all test targets
add_version_definitions(tst_settingsstoragetest)
add_version_definitions(tst_headerstoragetest) 
add_version_definitions(tst_blackchirpcsvtest)
add_version_definitions(tst_spcatparser)
add_version_definitions(tst_xiamparser)
add_version_definitions(tst_genericxyparser)
add_version_definitions(tst_overlayprocessing)
add_version_definitions(tst_unifiedoverlaydialog)
add_version_definitions(tst_overlayoperations)

# Simplified overlay operations test without heavy dependencies
add_executable(tst_overlayoperations_simple tests/tst_overlayoperations_simple.cpp
    src/gui/overlay/overlaytypespecificwidget.cpp
    src/data/processing/overlayprocessmanager.cpp
    src/data/processing/overlayoperation.cpp
    src/data/experiment/overlaybase.cpp
    src/data/experiment/overlaytypes.cpp
    src/data/experiment/catalogdata.cpp
    src/data/experiment/catalogparser.cpp
    src/data/experiment/fileparser.cpp
    src/data/experiment/catalogparserregistry.cpp
    src/data/analysis/ft.cpp
    src/data/analysis/analysis.cpp
    src/data/storage/settingsstorage.cpp
    src/data/experiment/fid.cpp)
add_test(NAME tst_overlayoperations_simple COMMAND tst_overlayoperations_simple)
target_link_libraries(tst_overlayoperations_simple PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)
add_version_definitions(tst_overlayoperations_simple)

# Minimal test for OperationCapability struct only (no heavy dependencies)
add_executable(tst_operation_capability_only tests/tst_operation_capability_only.cpp
    src/data/storage/settingsstorage.cpp)
add_test(NAME tst_operation_capability_only COMMAND tst_operation_capability_only)
target_link_libraries(tst_operation_capability_only PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)
add_version_definitions(tst_operation_capability_only)

# Background convolution execution test (end-to-end processing)
add_executable(tst_background_convolution tests/tst_background_convolution.cpp
    src/data/processing/overlayprocessmanager.cpp
    src/data/processing/overlayoperation.cpp
    src/data/experiment/overlaybase.cpp
    src/data/experiment/overlaytypes.cpp
    src/data/experiment/catalogdata.cpp
    src/data/experiment/catalogparser.cpp
    src/data/experiment/fileparser.cpp
    src/data/experiment/catalogparserregistry.cpp
    src/data/experiment/spcatparser.cpp
    src/data/experiment/xiamparser.cpp
    src/data/analysis/ft.cpp
    src/data/analysis/analysis.cpp
    src/data/storage/settingsstorage.cpp
    src/data/experiment/fid.cpp)
add_test(NAME tst_background_convolution COMMAND tst_background_convolution)
target_link_libraries(tst_background_convolution PRIVATE Qt6::Gui Qt6::Test Qt6::Widgets)
add_version_definitions(tst_background_convolution)
