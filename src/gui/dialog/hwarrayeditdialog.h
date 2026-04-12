#ifndef HWARRAYEDITDIALOG_H
#define HWARRAYEDITDIALOG_H

#include <QDialog>
#include <QStringList>
#include <data/storage/settingsstorage.h>

class QTableWidget;
class QPushButton;

/*!
 * \brief Sub-dialog for editing a hardware array setting
 *
 * Shows entries in a QTableWidget (rows = entries, columns = sub-keys).
 * Supports adding, removing, and reordering entries. All cell values are
 * edited as text and round-trip through QVariant on save.
 *
 * Opened from the Edit button in HwSettingsWidget array rows.
 */
class HwArrayEditDialog : public QDialog
{
    Q_OBJECT
public:
    /*!
     * \brief Construct the array edit dialog
     * \param label    Display name for the array (used as window title)
     * \param subKeys  Ordered list of sub-key names (column headers)
     * \param entries  Current entries to pre-populate the table
     * \param parent   Parent widget
     */
    HwArrayEditDialog(const QString &label,
                      const QStringList &subKeys,
                      std::vector<SettingsStorage::SettingsMap> entries,
                      QWidget *parent = nullptr);

    /*!
     * \brief Return the edited entries (only valid after accept())
     */
    std::vector<SettingsStorage::SettingsMap> result() const;

    QSize sizeHint() const override;

private slots:
    void addRow();
    void removeRow();
    void moveUp();
    void moveDown();
    void updateButtonStates();

private:
    QTableWidget *p_table;
    QPushButton *p_removeButton;
    QPushButton *p_upButton;
    QPushButton *p_downButton;

    QStringList d_subKeys;
    std::vector<SettingsStorage::SettingsMap> d_result;

    void accept() override;
};

#endif // HWARRAYEDITDIALOG_H
