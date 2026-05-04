#ifndef CUSTOMPROTOCOLWIDGET_H
#define CUSTOMPROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>
#include <hardware/core/hardwareregistry.h>

class QFormLayout;
class QLineEdit;
class QSpinBox;
class QPushButton;
class QVBoxLayout;

/*!
 * \brief Protocol widget for Custom communication protocol
 *
 * Dynamically generates input fields based on CustomCommDef descriptors retrieved
 * from the HardwareRegistry. Supports two construction paths:
 *
 * - Existing-profile mode (used by CommunicationDialog): pass \a hardwareKey in
 *   "type.label" format; the widget loads schema from the registry and values from
 *   the BC::Key::Comm::custom group in SettingsStorage.
 *
 * - New-profile mode (used by AddProfileDialog): pass \a hwType and \a hwImpl;
 *   the widget loads schema from the registry and initializes fields to their
 *   type-appropriate defaults. Call saveToStorage() on accept to persist the values.
 */
class CustomProtocolWidget : public ProtocolWidget
{
    Q_OBJECT

public:
    /*!
     * \brief Construct in existing-profile mode.
     * \param hardwareKey  Per-profile storage key in "type.label" format.
     * \param parent       Parent widget.
     */
    explicit CustomProtocolWidget(const QString& hardwareKey, QWidget *parent = nullptr);

    /*!
     * \brief Construct in new-profile mode (no on-disk profile yet).
     * \param hwType  Hardware type key (e.g., "IOBoard").
     * \param hwImpl  Implementation key (e.g., "LabjackU3").
     * \param parent  Parent widget.
     */
    explicit CustomProtocolWidget(const QString& hwType, const QString& hwImpl,
                                  QWidget *parent = nullptr);

    CommunicationProtocol::CommType getProtocolType() const override { return CommunicationProtocol::Custom; }
    void loadProtocolSettings() override;
    void saveProtocolSpecificSettings() override;

    /*!
     * \brief Write current field values to the BC::Key::Comm::custom group of \a storageKey.
     *
     * Used by AddProfileDialog on accept to persist custom-comm values for a newly created
     * profile before the hardware object is first constructed.
     */
    void saveToStorage(const QString& storageKey) const;

private:
    void setupUI();
    void generateDynamicUI(const QVector<CustomCommDef>& defs);
    void clearDynamicUI();

    QString d_hwImpl;

    QVBoxLayout *p_layout;
    QFormLayout *p_formLayout;

    QVector<QWidget*> d_dynamicWidgets;
    QVector<QPair<QString, QLineEdit*>> d_stringFields;    ///< key → line edit (String type)
    QVector<QPair<QString, QSpinBox*>> d_intFields;        ///< key → spin box  (Int type)
    QVector<QPair<QString, QLineEdit*>> d_filePathFields;  ///< key → line edit  (FilePath type)
};

#endif // CUSTOMPROTOCOLWIDGET_H
