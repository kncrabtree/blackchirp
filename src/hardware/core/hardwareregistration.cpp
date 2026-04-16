#include "hardwareregistration.h"
#include "hardwareregistry.h"

void initializeHardwareRegistrations()
{
    // Hardware registrations are performed automatically through static
    // HardwareAutoRegistration instances in each hardware implementation file.
    // This function serves as a centralized point for any additional
    // registration logic if needed in the future.

    // Force evaluation of any lazy registration by accessing the registry
    HardwareRegistry::instance();
}