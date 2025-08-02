#ifndef HARDWAREREGISTRATION_H
#define HARDWAREREGISTRATION_H

#include "hardwareregistry.h"
#include <QStringList>
#include <functional>

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
                           const QString& prettyName, const QString& description,
                           std::function<HardwareObject*()> factory)
    {
        HardwareRegistry::instance().registerHardware(
            key, subKey, prettyName, description, factory
        );
    }
};

/*!
 * \brief Register hardware implementation using introspection
 * 
 * This macro creates a temporary instance of the hardware class to extract
 * its keys, then registers it. This ensures the hardware object is the
 * single source of truth for its keys.
 * 
 * \param CLASS Hardware class name
 * \param NAME Pretty name for UI
 * \param DESC Description
 */
#define REGISTER_HARDWARE(CLASS, NAME, DESC) \
    static HardwareAutoRegistration register_##CLASS( \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_key; \
        }(), \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_subKey; \
        }(), \
        NAME, DESC, \
        []() -> HardwareObject* { return new CLASS(); } \
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