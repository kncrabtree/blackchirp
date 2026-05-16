#ifndef SETTINGSTABLE_H
#define SETTINGSTABLE_H

#include <QTableWidget>
#include <QHash>
#include <QList>

class QCheckBox;

/*!
 * \brief Compact two-column "Setting / Value" table shared across the
 *        hardware-settings and overlay-configuration UIs.
 *
 * Consolidates the makeSettingsTable/addTableRow idiom that was
 * previously hand-rolled in several widgets. The table is borderless,
 * non-selectable, read-only, sizes to its contents, and never shows a
 * vertical scrollbar.
 *
 * Rows come in three flavors:
 * - a label + value-widget row (single widget, or a pair laid out side
 *   by side in the value cell);
 * - a bold, spanned, theme-shaded section/heading row that replaces the
 *   nested QGroupBox titles of the old layout;
 * - a spanned checkable section row whose bound child rows collapse
 *   (via setRowHidden) when the box is unchecked, reproducing the old
 *   checkable-QGroupBox behavior.
 */
class SettingsTable : public QTableWidget
{
    Q_OBJECT
public:
    explicit SettingsTable(QWidget *parent = nullptr);

    /*!
     * \brief Append a label + single value widget.
     * \return the new row index.
     */
    int addSettingRow(const QString &label, QWidget *value,
                       const QString &tooltip = {});

    /*!
     * \brief Append a label + two widgets laid side by side in the
     *        value cell (e.g. checkbox + spinbox, input + button).
     * \return the new row index.
     */
    int addSettingRow(const QString &label, QWidget *first, QWidget *second,
                       const QString &tooltip = {});

    /*!
     * \brief Append a bold, spanned, theme-shaded heading row.
     * \return the new row index.
     */
    int addSectionRow(const QString &title);

    /*!
     * \brief Append a spanned heading row with a leading checkbox.
     *
     * Rows registered with bindSectionRows() are shown/hidden whenever
     * the box is toggled.
     *
     * \param outBox optional out-pointer to the created checkbox.
     * \return the new row index.
     */
    int addCheckableSectionRow(const QString &title, bool checked,
                               QCheckBox **outBox = nullptr);

    /*!
     * \brief Bind value rows to a checkable section row created earlier
     *        and apply the box's current state immediately.
     */
    void bindSectionRows(int sectionRow, const QList<int> &rows);

private:
    void applySectionShading(int row, QWidget *cellWidget = nullptr);

    /// Add \a extraHeight px to the enclosing top-level window so
    /// newly-shown section rows are not clipped. Grow-only by
    /// construction (called only on expand). No-op without a window.
    void growEnclosingWindow(int extraHeight);

    QHash<int, QCheckBox*> d_sectionBoxes;
};

#endif // SETTINGSTABLE_H
