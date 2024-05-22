#ifndef CUSTOMINSTRUMENT_H
#define CUSTOMINSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

/*!
 * Namespace for CustomInstrument communication details
 */
namespace BC::Key::Custom {
static const QString comm{"comm"}; /*!< Key for communication array. */
static const QString type{"type"}; /*!< Type of data entry field (stringKey or intKey). */
static const QString key{"key"}; /*!< Identifier key for this data field. */
static const QString intKey{"int"}; /*!< Key for integer data entry type. */
static const QString intMin{"min"}; /*!< Minimum allowed integer value. */
static const QString intMax{"max"}; /*!< Maximum allowed integer value. */
static const QString stringKey{"string"}; /*!< Key for string data entry type. */
static const QString maxLen{"length"}; /*!< Maximum allowed length of string. */
static const QString label{"name"}; /*!< Label displayed to user for this data field. */
}


/*!
 * \brief An instrument which has a non-QIODevice communication method
 * 
 * Similar to VirtualInstrument, this class leaves the device pointer null. Its
 * purpose is to provide keys that are used in the CommunicationDialog to
 * present the user with widgets for entering and information needed to
 * configure the device's communication. For example, the user may need to
 * specify a device path or serial number in order to initialize the object. By
 * assigning values to the keys defined in the BC::Key::Custom namespace in the
 * final HardwareObject implementation's constructor, the CommunicationDialog
 * will construct the UI and obtain values from the user.
 * 
 * This is done by first creating a SettingsStorage array with the key
 * BC::Key::Custom::comm. For each quantity that is needed from the user, add a
 * SettingsStorage::SettingsMap with, at minimum, definitions for
 * BC::Key::Custom::label, BC::Key::Custom::key, and BC::Key::Custom::type. As
 * an example, the followng code would add entries for a device path (string)
 * and serial number (integer >0). It is important to check for the existence
 * of the BC::Key::Custom::comm key first so that existing settings are not
 * overwritten.
 * 
 *     if(!containsArray(BC::Key::Custom::comm))
 *         setArray(BC::Key::Custom::comm, {
 *                   {{BC::Key::Custom::key,"devPath"},
 *                    {BC::Key::Custom::type,BC::Key::Custom::stringKey},
 *                    {BC::Key::Custom::label,"Device Path"}},
 *                   {{BC::Key::Custom::key,"serialNo"},
 *                    {BC::Key::Custom::type,BC::Key::Custom::intKey},
 *                    {BC::Key::Custom::label,"Serial Number"},
 *                    {BC::Key::Custom::intMin,0}}
 *                });
 * 
 */
class CustomInstrument : public CommunicationProtocol
{
public:
    explicit CustomInstrument(QString key, QObject *parent = nullptr);

public slots:
    void initialize() override;
    bool testConnection() override;
};

#endif // CUSTOMINSTRUMENT_H
