#ifndef SETTINGSTABLE_H
#define SETTINGSTABLE_H

#include <QTableWidget>
#include <QHash>
#include <QList>

class QCheckBox;
class QLabel;

/*!
 * \brief Compact, borderless two-column "Setting / Value" table.
 *
 * Non-selectable, read-only, sizes to its contents, and never shows a
 * vertical scrollbar. Rows come in three flavors:
 * - a label + value-widget row (single widget, or a pair laid out side
 *   by side in the value cell);
 * - a bold, spanned, theme-shaded section/heading row;
 * - a spanned checkable section row whose bound child rows collapse
 *   (via setRowHidden) when the box is unchecked.
 *
 * A checkable section row can be retitled, switched between a checkbox
 * and a plain centered heading, and have its bound rows enabled or
 * disabled without hiding them, so one row can serve both a
 * non-checkable and a checkable state. The backing QCheckBox is created
 * once and outlives every mode change, so external connections to it
 * survive.
 */
class SettingsTable : public QTableWidget
{
    Q_OBJECT
public:
    explicit SettingsTable(QWidget *parent = nullptr);

    /*!
     * \brief Floor the vertical minimum at the content height.
     *
     * The table never shows a vertical scrollbar, so it must never be
     * laid out shorter than its (visible) rows or the bottom rows are
     * silently clipped. This matters inside a QMainWindow dock area,
     * where a panel shown next to an existing one is otherwise pinned
     * at the widget minimum. Width still defers to the base class.
     */
    QSize minimumSizeHint() const override;

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
     *
     * Rendered through the same centered cell-widget mechanism as a
     * checkable section row (a plain QLabel instead of a QCheckBox), so
     * the band color is identical for checkable and non-checkable
     * headings. The row is tracked like a checkable section (with a
     * null checkbox) so it can be retitled, have rows bound to it, and
     * be shown/hidden as a unit.
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
     * \brief Enable or disable a section's bound rows (and the section
     *        heading itself) without changing visibility — the disabled
     *        counterpart of the hide-on-uncheck collapse.
     */
    void setBoundRowsEnabled(int sectionRow, bool enabled);

    /*!
     * \brief Show or hide a whole section as a unit: the heading row
     *        and every bound row.
     *
     * Collapse-aware: when re-showing, a bound row that also belongs to
     * a nested checkable section whose box is unchecked stays hidden,
     * so a plain container section can wrap nested collapsible
     * sub-sections without fighting their collapse state. Does not grow
     * the enclosing window (used for programmatic context switches, not
     * user toggles).
     */
    void setSectionVisible(int sectionRow, bool visible);

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
        /// One-shot: the enclosing window is grown on the *first*
        /// expand only, and only when the section started collapsed.
        /// Cleared once consumed so repeated toggling never re-grows.
        bool growPending = false;
    };

    /// Paint the section-heading band on the row's QTableWidgetItem
    /// (AlternateBase fill + EmphasisText, bold, centered). Both plain
    /// and checkable headings go through this single mechanism so the
    /// band is byte-for-byte identical; for a checkable row the
    /// transparent cell widget sits on top of this item.
    void styleSectionItem(int row);

    /// Emphasize a heading's visible text widget (the QLabel or
    /// QCheckBox that sits over the band) to match the item styling.
    void styleSectionText(QWidget *textWidget);

    /// Add \a extraHeight px to the enclosing top-level window so
    /// newly-shown section rows are not clipped. Grow-only by
    /// construction (called only on expand). No-op without a window.
    void growEnclosingWindow(int extraHeight);

    QHash<int, Section> d_sections;
};

#endif // SETTINGSTABLE_H
