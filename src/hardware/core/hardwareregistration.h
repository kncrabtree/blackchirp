#ifndef HARDWAREREGISTRATION_H
#define HARDWAREREGISTRATION_H

#include "hardwareregistry.h"
#include <QStringList>
#include <functional>
#include <QMetaObject>

class HardwareObject;

/*!
 * \brief Hardware registration helper macros and utilities
 * 
 * This file provides convenience macros and functions to simplify hardware
 * registration with the HardwareRegistry. It enables automatic registration
 * of hardware implementations during program startup.
 */

/*!
 * \brief Auto-registration helper class
 * 
 * This class performs hardware registration in its constructor, allowing
 * for automatic registration when a static instance is created.
 */
class HardwareAutoRegistration
{
public:
    HardwareAutoRegistration(const QString& key, const QString& subKey, 
                           const QString& description,
                           std::function<HardwareObject*(const QString&)> factory)
    {
        HardwareRegistry::instance().registerHardware(
            key, subKey, description, factory
        );
    }
};

/*!
 * \brief Helper function to find the hardware base type
 * 
 * Traverses the inheritance hierarchy to find the direct child of HardwareObject
 */
inline QString findHardwareBaseType(const QMetaObject* metaObj) {
    const QMetaObject* current = metaObj;
    const QMetaObject* parent = current->superClass();
    
    // Traverse up until we find HardwareObject as the parent
    while (parent && QString(parent->className()) != "HardwareObject") {
        current = parent;
        parent = current->superClass();
    }
    
    // Return the direct child of HardwareObject
    return QString(current->className());
}

/*!
 * \brief Register hardware implementation using Qt metaobjects
 * 
 * This macro uses Qt's metaobject system to automatically derive hardware
 * type and implementation keys from class names, eliminating the need for
 * manual key management and preventing static counter increments.
 * 
 * \param CLASS Hardware class name
 * \param DESC Description (should include manufacturer and model)
 */
#define REGISTER_HARDWARE_META(CLASS, DESC) \
    static HardwareAutoRegistration register_##CLASS( \
        findHardwareBaseType(&CLASS::staticMetaObject), \
        QString(CLASS::staticMetaObject.className()), \
        DESC, \
        [](const QString& label) -> HardwareObject* { \
            return new CLASS(label); \
        } \
    );

/*!
 * \brief Register library dependency for hardware implementation
 * 
 * This macro adds a vendor library dependency to an already registered hardware
 * implementation. It should be used after REGISTER_HARDWARE_META in hardware
 * implementation files where the necessary library includes are available.
 * 
 * \param CLASS Hardware class name (must already be registered)
 * \param LIBRARY_NAME Library class name (e.g., SpectrumLibrary, LabjackLibrary)
 */
#define REGISTER_LIBRARY(CLASS, LIBRARY_NAME) \
    static bool library_registered_##CLASS##_##LIBRARY_NAME = \
        HardwareRegistry::instance().addLibraryDependency( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            #LIBRARY_NAME, \
            []() -> VendorLibrary* { return &LIBRARY_NAME::instance(); } \
        );

/*!
 * \brief Register supported communication protocols for a hardware implementation
 *
 * This macro registers the supported protocols in the HardwareRegistry so they
 * are available without instantiating the hardware object (avoiding vtable issues
 * in the constructor). Should be placed immediately after REGISTER_HARDWARE_META.
 *
 * \param CLASS Hardware class name (must already be registered)
 * \param ... One or more CommunicationProtocol::CommType values
 */
#define REGISTER_HARDWARE_PROTOCOLS(CLASS, ...) \
    static bool protocols_registered_##CLASS = \
        HardwareRegistry::instance().addSupportedProtocols( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            QVector<CommunicationProtocol::CommType>{__VA_ARGS__} \
        );

/*!
 * \brief Register configuration parameters for a hardware implementation
 *
 * This macro registers parameters that require UI input before the hardware
 * object can be constructed (e.g., numChannels, tunable). The CLASS must
 * define a static method: static QVector<HwConfigParam> configParams();
 *
 * Should be placed after REGISTER_HARDWARE_META in the .cpp file.
 *
 * \param CLASS Hardware class name (must already be registered and have configParams())
 */
#define REGISTER_HARDWARE_PARAMS(CLASS) \
    static bool config_params_registered_##CLASS = \
        HardwareRegistry::instance().addConfigParams( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            CLASS::configParams() \
        );

/*!
 * \brief Register scalar settings for a hardware class
 *
 * Registers setting definitions with metadata (labels, descriptions, priorities)
 * in the HardwareRegistry. Settings are available before object construction
 * and presented at profile creation time.
 *
 * \param CLASS Hardware class name (must already be registered)
 * \param ... HwSettingDef initializer lists: {key, "Label", "Description", defaultVal, min, max, priority}
 */
#define REGISTER_HARDWARE_SETTINGS(CLASS, ...) \
    static bool settings_registered_##CLASS = \
        HardwareRegistry::instance().addSettingDefs( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            {__VA_ARGS__} \
        );

/*!
 * \brief Register array setting metadata (call once per array key)
 *
 * Declares an array-type setting with display metadata. Must be called
 * before any REGISTER_HARDWARE_ARRAY_ENTRY calls for the same array key.
 *
 * \param CLASS Hardware class name (must already be registered)
 * \param ARRAY_KEY String constant for the array key
 * \param LABEL User-facing display label
 * \param DESC Explanatory description/tooltip
 * \param PRIORITY HwSettingPriority value
 */
#define REGISTER_HARDWARE_ARRAY(CLASS, ARRAY_KEY, LABEL, DESC, PRIORITY) \
    static bool BC_ARRDEF_VAR(CLASS, ARRAY_KEY) = \
        HardwareRegistry::instance().addArraySettingDef( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            ARRAY_KEY, LABEL, DESC, PRIORITY);

/*!
 * \brief Register one entry in an array setting (call once per entry)
 *
 * Adds a single entry to a previously declared array setting.
 * Each entry is a SettingsStorage::SettingsMap (std::map<QString,QVariant>).
 *
 * \param CLASS Hardware class name
 * \param ARRAY_KEY String constant for the array key (must match a prior REGISTER_HARDWARE_ARRAY)
 * \param ... Key-value pairs: {{subKey1, value1}, {subKey2, value2}}
 */
#define REGISTER_HARDWARE_ARRAY_ENTRY(CLASS, ARRAY_KEY, ...) \
    static bool BC_ARRENTRY_VAR(CLASS, __COUNTER__) = \
        HardwareRegistry::instance().addArraySettingEntry( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            ARRAY_KEY, \
            SettingsStorage::SettingsMap{__VA_ARGS__} \
        );

// Helpers for unique static variable names in array macros
#define BC_ARRDEF_CONCAT(a, b) a##b
#define BC_ARRDEF_VAR(CLASS, KEY) BC_ARRDEF_CONCAT(arraydef_##CLASS##_, __LINE__)
#define BC_ARRENTRY_CONCAT(a, b) a##b
#define BC_ARRENTRY_VAR(CLASS, N) BC_ARRENTRY_CONCAT(arrayentry_##CLASS##_, N)

/*!
 * \brief Register hardware implementation using introspection (legacy)
 * 
 * This macro creates a temporary instance of the hardware class to extract
 * its keys, then registers it. This ensures the hardware object is the
 * single source of truth for its keys.
 * 
 * \param CLASS Hardware class name
 * \param DESC Description (should include manufacturer and model)
 */
#define REGISTER_HARDWARE(CLASS, DESC) \
    static HardwareAutoRegistration register_##CLASS( \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_key; \
        }(), \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_model; \
        }(), \
        DESC, \
        [](const QString& label) -> HardwareObject* { return new CLASS(label); } \
    );


/*!
 * \brief Initialize all hardware registrations
 * 
 * This function should be called during program startup to ensure
 * all hardware implementations are registered. It provides a
 * centralized place to manage hardware registration order if needed.
 */
void initializeHardwareRegistrations();

#endif // HARDWAREREGISTRATION_H
