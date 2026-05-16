#ifndef SETTINGSTABLE_H
#define SETTINGSTABLE_H

#include <QTableWidget>
#include <QHash>
#include <QList>

class QCheckBox;
class QLabel;

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
 *
 * A checkable section row can be retitled, switched between a checkbox
 * and a plain centered heading, and have its bound rows enabled or
 * disabled without hiding them, so a single row can stand in for the
 * Creation (non-checkable) and Settings (checkable) states of the
 * overlay source-file-configuration QGroupBox it replaces. The checkbox
 * object is created once and outlives every mode change, so external
 * connections to it survive.
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

    /*!
     * \brief Retitle a section row in place (works for both the
     *        checkbox and the plain-heading rendering).
     */
    void setSectionTitle(int sectionRow, const QString &title);

    /*!
     * \brief Switch a checkable section row between a leading checkbox
     *        and a plain centered heading.
     *
     * The underlying QCheckBox is kept alive across the change (only
     * the displayed cell content swaps), so connections to it and the
     * bound-row wiring survive. A non-checkable section never collapses;
     * its bound rows' visibility is left to the caller / bound-row
     * machinery (a plain heading does not itself hide anything).
     */
    void setSectionCheckable(int sectionRow, bool checkable);

    /*!
     * \brief Enable or disable a section's bound rows without changing
     *        their visibility (the disabled counterpart of the
     *        hide-on-uncheck collapse).
     */
    void setBoundRowsEnabled(int sectionRow, bool enabled);

    /*!
     * \brief Re-apply the hidden state of a section's bound rows from
     *        the checkbox's current state, without growing the window.
     *
     * Used after a signal-blocked programmatic setChecked() so the
     * collapse stays consistent without the user-toggle window growth.
     */
    void applySectionVisibility(int sectionRow);

    /*!
     * \brief The checkbox backing a checkable section row, or nullptr.
     *
     * Stable across setSectionCheckable() mode changes.
     */
    QCheckBox *sectionCheckBox(int sectionRow) const;

private:
    struct Section {
        QCheckBox *box = nullptr;       ///< always alive, even when plain
        QWidget *wrap = nullptr;        ///< centered cell host
        QLabel *plainLabel = nullptr;   ///< shown when non-checkable
        QString title;
        bool checkable = true;
        QList<int> boundRows;
    };

    void applySectionShading(int row, QWidget *cellWidget = nullptr);

    /// Add \a extraHeight px to the enclosing top-level window so
    /// newly-shown section rows are not clipped. Grow-only by
    /// construction (called only on expand). No-op without a window.
    void growEnclosingWindow(int extraHeight);

    QHash<int, Section> d_sections;
};

#endif // SETTINGSTABLE_H
