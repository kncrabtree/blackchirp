#include "settingstable.h"

#include <QHeaderView>
#include <QHBoxLayout>
#include <QCheckBox>

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

    // Center the checkbox within the spanned section cell.
    auto *wrap = new QWidget(this);
    auto *hbl = new QHBoxLayout(wrap);
    hbl->setContentsMargins(0, 0, 0, 0);
    hbl->addStretch(1);
    hbl->addWidget(box);
    hbl->addStretch(1);
    setCellWidget(row, 0, wrap);
    applySectionShading(row, wrap);

    d_sectionBoxes.insert(row, box);
    if (outBox)
        *outBox = box;
    return row;
}

void SettingsTable::bindSectionRows(int sectionRow, const QList<int> &rows)
{
    auto it = d_sectionBoxes.constFind(sectionRow);
    if (it == d_sectionBoxes.constEnd())
        return;

    QCheckBox *box = it.value();
    auto apply = [this, rows](bool on) {
        for (int r : rows)
            setRowHidden(r, !on);
    };
    connect(box, &QCheckBox::toggled, this, apply);
    apply(box->isChecked());
}
