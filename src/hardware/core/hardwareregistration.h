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
                           const QStringList& dependencies,
                           std::function<HardwareObject*()> factory,
                           std::function<bool()> availabilityCheck,
                           bool required = false)
    {
        HardwareRegistry::instance().registerHardware(
            key, subKey, prettyName, description, dependencies,
            factory, availabilityCheck, required
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
 * \param DEPS Dependencies (as QStringList)
 * \param AVAIL_CHECK Availability check function
 * \param REQUIRED Whether hardware is required (default: false)
 */
#define REGISTER_HARDWARE(CLASS, NAME, DESC, DEPS, AVAIL_CHECK, REQUIRED) \
    static HardwareAutoRegistration register_##CLASS( \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_key; \
        }(), \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_subKey; \
        }(), \
        NAME, DESC, DEPS, \
        []() -> HardwareObject* { return new CLASS(); }, \
        AVAIL_CHECK, REQUIRED \
    );

/*!
 * \brief Register hardware with vendor library dependency
 * 
 * Convenience macro for hardware that depends on a vendor library.
 * Keys are extracted from the hardware object itself.
 * 
 * \param CLASS Hardware class name
 * \param NAME Pretty name for UI
 * \param DESC Description
 * \param LIBRARY_CLASS Vendor library class name
 * \param REQUIRED Whether hardware is required (default: false)
 */
#define REGISTER_HARDWARE_WITH_LIBRARY(CLASS, NAME, DESC, LIBRARY_CLASS, REQUIRED) \
    static HardwareAutoRegistration register_##CLASS( \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_key; \
        }(), \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_subKey; \
        }(), \
        NAME, DESC, {#LIBRARY_CLASS}, \
        []() -> HardwareObject* { return new CLASS(); }, \
        []() -> bool { return LIBRARY_CLASS::instance().isAvailable(); }, \
        REQUIRED \
    );

/*!
 * \brief Register simple hardware without dependencies
 * 
 * Convenience macro for hardware that has no external dependencies.
 * Keys are extracted from the hardware object itself.
 * 
 * \param CLASS Hardware class name
 * \param NAME Pretty name for UI
 * \param DESC Description
 * \param REQUIRED Whether hardware is required (default: false)
 */
#define REGISTER_SIMPLE_HARDWARE(CLASS, NAME, DESC, REQUIRED) \
    static HardwareAutoRegistration register_##CLASS( \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_key; \
        }(), \
        []() { \
            auto temp = std::unique_ptr<CLASS>(new CLASS()); \
            return temp->d_subKey; \
        }(), \
        NAME, DESC, {}, \
        []() -> HardwareObject* { return new CLASS(); }, \
        []() -> bool { return true; }, \
        REQUIRED \
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