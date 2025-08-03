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
            return temp->d_subKey; \
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
