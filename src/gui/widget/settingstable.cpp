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
    if (first)
        hbl->addWidget(first);
    if (second)
        hbl->addWidget(second);
    hbl->addStretch(1);
    return addSettingRow(label, cell, tooltip);
}

void SettingsTable::applySectionShading(int row, QWidget *cellWidget)
{
    const QColor band = palette().color(QPalette::AlternateBase);
    const QColor fg = ThemeColors::getThemeAwareColor(ThemeColors::EmphasisText, this);

    if (cellWidget) {
        cellWidget->setAutoFillBackground(true);
        QPalette pal = cellWidget->palette();
        pal.setColor(QPalette::Window, band);
        pal.setColor(QPalette::WindowText, fg);
        cellWidget->setPalette(pal);
        QFont f = cellWidget->font();
        f.setBold(true);
        cellWidget->setFont(f);
    } else if (auto *item = this->item(row, 0)) {
        item->setBackground(band);
        item->setForeground(fg);
        item->setTextAlignment(Qt::AlignCenter);
        QFont f = item->font();
        f.setBold(true);
        item->setFont(f);
    }
}

int SettingsTable::addSectionRow(const QString &title)
{
    int row = rowCount();
    insertRow(row);
    setSpan(row, 0, 1, 2);

    auto *item = new QTableWidgetItem(title);
    item->setFlags(Qt::ItemIsEnabled);
    setItem(row, 0, item);
    applySectionShading(row);
    return row;
}

int SettingsTable::addCheckableSectionRow(const QString &title, bool checked,
                                          QCheckBox **outBox)
{
    int row = rowCount();
    insertRow(row);
    setSpan(row, 0, 1, 2);

    auto *box = new QCheckBox(title, this);
    box->setChecked(checked);

    // The plain-heading rendering used when the section is made
    // non-checkable. Created up front and kept hidden so switching
    // modes never recreates a widget the row index depends on.
    auto *plain = new QLabel(title, this);
    plain->setVisible(false);

    // Center the active element within the spanned section cell.
    auto *wrap = new QWidget(this);
    auto *hbl = new QHBoxLayout(wrap);
    hbl->setContentsMargins(0, 0, 0, 0);
    hbl->addStretch(1);
    hbl->addWidget(box);
    hbl->addWidget(plain);
    hbl->addStretch(1);
    setCellWidget(row, 0, wrap);
    applySectionShading(row, wrap);

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

    // Initial state: hide collapsed rows without resizing (the window is
    // not yet shown during setup).
    for (int r : rows)
        setRowHidden(r, !box->isChecked());

    // On user toggle, reveal/hide the rows. When revealing, grow the
    // window by exactly the height of the rows now shown so they are not
    // clipped behind the (suppressed) scrollbar. Computed from the row
    // sizes rather than a deferred sizeHint, which is stale on the first
    // toggle because the table has not relaid out yet.
    connect(box, &QCheckBox::toggled, this, [this, rows](bool on) {
        int extra = 0;
        for (int r : rows) {
            setRowHidden(r, !on);
            if (on)
                extra += rowHeight(r) + 1; // row + its grid line
        }
        if (on && extra > 0)
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
