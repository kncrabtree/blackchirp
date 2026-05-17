#include "settingstable.h"

#include <QHeaderView>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QLabel>

#include <gui/style/themecolors.h>

SettingsTable::SettingsTable(QWidget *parent)
    : QTableWidget(0, 2, parent)
{
    horizontalHeader()->setStretchLastSection(true);
    horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    horizontalHeader()->setVisible(false);
    verticalHeader()->setVisible(false);
    setSelectionMode(QAbstractItemView::NoSelection);
    setEditTriggers(QAbstractItemView::NoEditTriggers);
    setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

QSize SettingsTable::minimumSizeHint() const
{
    // sizeHint() already tracks the visible content height through
    // AdjustToContents; adopt it as the vertical floor so the dock
    // layout cannot squeeze rows off the bottom.
    QSize base = QTableWidget::minimumSizeHint();
    return QSize(base.width(), sizeHint().height());
}

int SettingsTable::addSettingRow(const QString &label, QWidget *value,
                                 const QString &tooltip)
{
    int row = rowCount();
    insertRow(row);

    auto *labelItem = new QTableWidgetItem(label);
    labelItem->setFlags(Qt::ItemIsEnabled);
    if (!tooltip.isEmpty())
        labelItem->setToolTip(tooltip);
    setItem(row, 0, labelItem);

    if (value) {
        setCellWidget(row, 1, value);
        setRowHeight(row, value->sizeHint().height() + 4);
    }
    return row;
}

int SettingsTable::addSettingRow(const QString &label, QWidget *first,
                                 QWidget *second, const QString &tooltip)
{
    auto *cell = new QWidget(this);
    auto *hbl = new QHBoxLayout(cell);
    hbl->setContentsMargins(0, 0, 0, 0);

    // A horizontally-expanding widget (e.g. a QLineEdit path field)
    // fills the value cell; a fixed widget (e.g. a browse button) keeps
    // its hint. Only when nothing wants to expand do we add a trailing
    // spacer so the pair stays left-aligned, as before.
    auto expands = [](QWidget *w) {
        return w && (w->sizePolicy().horizontalPolicy()
                     & QSizePolicy::ExpandFlag);
    };
    bool any = expands(first) || expands(second);
    if (first)
        hbl->addWidget(first, expands(first) ? 1 : 0);
    if (second)
        hbl->addWidget(second, expands(second) ? 1 : 0);
    if (!any)
        hbl->addStretch(1);
    return addSettingRow(label, cell, tooltip);
}

void SettingsTable::styleSectionItem(int row)
{
    auto *it = item(row, 0);
    if (!it)
        return;
    it->setBackground(palette().color(QPalette::AlternateBase));
    it->setForeground(ThemeColors::getThemeAwareColor(
        ThemeColors::EmphasisText, this));
    it->setTextAlignment(Qt::AlignCenter);
    QFont f = it->font();
    f.setBold(true);
    it->setFont(f);
}

void SettingsTable::styleSectionText(QWidget *textWidget)
{
    if (!textWidget)
        return;
    const QColor fg = ThemeColors::getThemeAwareColor(
        ThemeColors::EmphasisText, this);
    QPalette pal = textWidget->palette();
    pal.setColor(QPalette::WindowText, fg);
    pal.setColor(QPalette::Text, fg);
    pal.setColor(QPalette::ButtonText, fg);
    textWidget->setPalette(pal);
    QFont f = textWidget->font();
    f.setBold(true);
    textWidget->setFont(f);
}

int SettingsTable::addSectionRow(const QString &title)
{
    int row = rowCount();
    insertRow(row);
    setSpan(row, 0, 1, 2);

    // Plain heading: a styled QTableWidgetItem (no cell widget). The
    // band comes from the item background — the original, correct
    // rendering that the checkable variant must now match.
    auto *it = new QTableWidgetItem(title);
    it->setFlags(Qt::ItemIsEnabled);
    setItem(row, 0, it);
    styleSectionItem(row);

    Section s;
    s.box = nullptr;
    s.wrap = nullptr;
    s.plainLabel = nullptr;
    s.title = title;
    s.checkable = false;
    d_sections.insert(row, s);
    return row;
}

int SettingsTable::addCheckableSectionRow(const QString &title, bool checked,
                                          QCheckBox **outBox)
{
    int row = rowCount();
    insertRow(row);
    setSpan(row, 0, 1, 2);

    // Background item carries the band, exactly as for a plain heading,
    // so the two render identically. The centering cell widget below is
    // transparent and sits on top of it.
    auto *bg = new QTableWidgetItem();
    bg->setFlags(Qt::ItemIsEnabled);
    setItem(row, 0, bg);
    styleSectionItem(row);

    auto *box = new QCheckBox(title, this);
    box->setChecked(checked);
    styleSectionText(box);

    // The plain-heading rendering used when the section is made
    // non-checkable. Created up front and kept hidden so switching
    // modes never recreates a widget the row index depends on.
    auto *plain = new QLabel(title, this);
    plain->setVisible(false);
    styleSectionText(plain);

    // Transparent centering host: no autoFillBackground / Window color,
    // so the band painted by the item shows through.
    auto *wrap = new QWidget(this);
    auto *hbl = new QHBoxLayout(wrap);
    hbl->setContentsMargins(0, 0, 0, 0);
    hbl->addStretch(1);
    hbl->addWidget(box);
    hbl->addWidget(plain);
    hbl->addStretch(1);
    setCellWidget(row, 0, wrap);

    Section s;
    s.box = box;
    s.wrap = wrap;
    s.plainLabel = plain;
    s.title = title;
    s.checkable = true;
    d_sections.insert(row, s);

    if (outBox)
        *outBox = box;
    return row;
}

void SettingsTable::bindSectionRows(int sectionRow, const QList<int> &rows)
{
    auto it = d_sections.find(sectionRow);
    if (it == d_sections.end())
        return;

    it->boundRows = rows;
    QCheckBox *box = it->box;

    // A plain (non-checkable) container section never collapses on its
    // own and must not disturb the visibility state any nested
    // collapsible sub-section already applied to these rows; just
    // record the membership. Whole-section show/hide is driven
    // explicitly via setSectionVisible().
    if (!box)
        return;

    // Initial state: hide collapsed rows without resizing (the window is
    // not yet shown during setup). The window is grown on the first
    // expand only, and only if the section started collapsed — a
    // section that starts expanded already has its rows accounted for
    // in the initial size, and repeated toggling must not keep growing.
    const bool startedCollapsed = !box->isChecked();
    it->growPending = startedCollapsed;
    for (int r : rows)
        setRowHidden(r, startedCollapsed);

    // On user toggle, reveal/hide the rows. The one-shot grow uses the
    // revealed rows' exact pixel height (computed from the row sizes,
    // not a deferred sizeHint, which is stale before the table relays
    // out) so they are not clipped behind the suppressed scrollbar.
    connect(box, &QCheckBox::toggled, this, [this, rows, sectionRow](bool on) {
        for (int r : rows)
            setRowHidden(r, !on);
        if (!on)
            return;
        auto s = d_sections.find(sectionRow);
        if (s == d_sections.end() || !s->growPending)
            return;
        s->growPending = false; // consume: grow exactly once
        int extra = 0;
        for (int r : rows)
            extra += rowHeight(r) + 1; // row + its grid line
        if (extra > 0)
            growEnclosingWindow(extra);
    });
}

void SettingsTable::setSectionTitle(int sectionRow, const QString &title)
{
    auto it = d_sections.find(sectionRow);
    if (it == d_sections.end())
        return;
    it->title = title;
    if (it->box)
        it->box->setText(title);
    if (it->plainLabel)
        it->plainLabel->setText(title);
    // Plain (item-based) heading: the title lives on the item itself.
    if (!it->box && !it->plainLabel) {
        if (auto *cell = item(sectionRow, 0))
            cell->setText(title);
    }
}

void SettingsTable::setSectionCheckable(int sectionRow, bool checkable)
{
    auto it = d_sections.find(sectionRow);
    if (it == d_sections.end() || it->checkable == checkable)
        return;

    it->checkable = checkable;
    // Swap only the displayed element; the QCheckBox stays alive so
    // bound-row and relay connections survive the mode change.
    if (it->box)
        it->box->setVisible(checkable);
    if (it->plainLabel)
        it->plainLabel->setVisible(!checkable);

    // A plain heading never collapses: its bound rows are always
    // expanded. (Any subclass-managed dynamic row visibility is
    // re-asserted by the caller afterwards.) Switching back to a
    // checkbox re-applies the collapse from the box's current state.
    if (!checkable) {
        for (int r : it->boundRows)
            setRowHidden(r, false);
    } else {
        const bool show = it->box && it->box->isChecked();
        for (int r : it->boundRows)
            setRowHidden(r, !show);
    }
}

void SettingsTable::setBoundRowsEnabled(int sectionRow, bool enabled)
{
    auto it = d_sections.constFind(sectionRow);
    if (it == d_sections.constEnd())
        return;

    // Grey out the heading too, matching the old whole-QGroupBox
    // disable this stands in for: a checkable heading via its cell
    // widget, a plain (item-based) heading via the item's flags.
    if (it->wrap) {
        it->wrap->setEnabled(enabled);
    } else if (auto *head = item(sectionRow, 0)) {
        Qt::ItemFlags f = head->flags();
        if (enabled)
            f |= Qt::ItemIsEnabled;
        else
            f &= ~Qt::ItemIsEnabled;
        head->setFlags(f);
    }

    for (int r : it->boundRows) {
        if (auto *labelItem = item(r, 0)) {
            Qt::ItemFlags f = labelItem->flags();
            if (enabled)
                f |= Qt::ItemIsEnabled;
            else
                f &= ~Qt::ItemIsEnabled;
            labelItem->setFlags(f);
        }
        if (auto *w = cellWidget(r, 0))
            w->setEnabled(enabled);
        if (auto *w = cellWidget(r, 1))
            w->setEnabled(enabled);
    }
}

void SettingsTable::setSectionVisible(int sectionRow, bool visible)
{
    auto it = d_sections.constFind(sectionRow);
    if (it == d_sections.constEnd())
        return;

    setRowHidden(sectionRow, !visible);

    for (int r : it->boundRows) {
        bool hide = !visible;
        if (visible) {
            // Defer to any nested checkable section that currently
            // collapses this row, so a plain container section can
            // wrap collapsible sub-sections.
            for (auto s = d_sections.constBegin();
                 s != d_sections.constEnd(); ++s) {
                if (s->box && !s->box->isChecked()
                    && s->boundRows.contains(r)) {
                    hide = true;
                    break;
                }
            }
        }
        setRowHidden(r, hide);
    }
}

void SettingsTable::applySectionVisibility(int sectionRow)
{
    auto it = d_sections.constFind(sectionRow);
    if (it == d_sections.constEnd() || !it->box)
        return;

    const bool show = it->box->isChecked();
    for (int r : it->boundRows)
        setRowHidden(r, !show);
}

QCheckBox *SettingsTable::sectionCheckBox(int sectionRow) const
{
    auto it = d_sections.constFind(sectionRow);
    if (it == d_sections.constEnd())
        return nullptr;
    return it->box;
}

void SettingsTable::growEnclosingWindow(int extraHeight)
{
    QWidget *w = window();
    if (!w || w == this || extraHeight <= 0)
        return;

    // Grow-only: we only ever add height, never shrink a dialog the
    // user may have enlarged.
    w->resize(w->width(), w->height() + extraHeight);
}
