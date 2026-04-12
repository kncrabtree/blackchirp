#include "hwarrayeditdialog.h"

#include <QTableWidget>
#include <QHeaderView>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>

HwArrayEditDialog::HwArrayEditDialog(const QString &label,
                                     const QStringList &subKeys,
                                     std::vector<SettingsStorage::SettingsMap> entries,
                                     QWidget *parent)
    : QDialog(parent), d_subKeys(subKeys), d_result(entries)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(QString("Edit: %1").arg(label));
    setModal(true);

    auto *vbl = new QVBoxLayout(this);

    // Table
    p_table = new QTableWidget(0, d_subKeys.size(), this);
    p_table->setHorizontalHeaderLabels(d_subKeys);
    p_table->horizontalHeader()->setStretchLastSection(true);
    p_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    p_table->verticalHeader()->setVisible(false);
    p_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_table->setSelectionMode(QAbstractItemView::SingleSelection);

    // Populate table from entries
    for (const auto &entry : entries) {
        int row = p_table->rowCount();
        p_table->insertRow(row);
        for (int col = 0; col < d_subKeys.size(); ++col) {
            auto it = entry.find(d_subKeys.at(col));
            QString text = (it != entry.end()) ? it->second.toString() : QString();
            p_table->setItem(row, col, new QTableWidgetItem(text));
        }
    }

    connect(p_table, &QTableWidget::itemSelectionChanged,
            this, &HwArrayEditDialog::updateButtonStates);

    vbl->addWidget(p_table, 1);

    // Row manipulation buttons
    auto *hbl = new QHBoxLayout;

    auto *addButton = new QPushButton("Add", this);
    connect(addButton, &QPushButton::clicked, this, &HwArrayEditDialog::addRow);
    hbl->addWidget(addButton);

    p_removeButton = new QPushButton("Remove", this);
    p_removeButton->setEnabled(false);
    connect(p_removeButton, &QPushButton::clicked, this, &HwArrayEditDialog::removeRow);
    hbl->addWidget(p_removeButton);

    hbl->addStretch(1);

    p_upButton = new QPushButton("Move Up", this);
    p_upButton->setEnabled(false);
    connect(p_upButton, &QPushButton::clicked, this, &HwArrayEditDialog::moveUp);
    hbl->addWidget(p_upButton);

    p_downButton = new QPushButton("Move Down", this);
    p_downButton->setEnabled(false);
    connect(p_downButton, &QPushButton::clicked, this, &HwArrayEditDialog::moveDown);
    hbl->addWidget(p_downButton);

    vbl->addLayout(hbl);

    // OK / Cancel
    auto *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(bb, &QDialogButtonBox::accepted, this, &HwArrayEditDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);
    vbl->addWidget(bb);
}

std::vector<SettingsStorage::SettingsMap> HwArrayEditDialog::result() const
{
    return d_result;
}

QSize HwArrayEditDialog::sizeHint() const
{
    return {500, 400};
}

void HwArrayEditDialog::addRow()
{
    int row = p_table->rowCount();
    p_table->insertRow(row);
    for (int col = 0; col < d_subKeys.size(); ++col)
        p_table->setItem(row, col, new QTableWidgetItem(QString()));
    p_table->selectRow(row);
}

void HwArrayEditDialog::removeRow()
{
    int row = p_table->currentRow();
    if (row >= 0)
        p_table->removeRow(row);
    updateButtonStates();
}

void HwArrayEditDialog::moveUp()
{
    int row = p_table->currentRow();
    if (row <= 0)
        return;

    for (int col = 0; col < p_table->columnCount(); ++col) {
        auto *above = p_table->takeItem(row - 1, col);
        auto *current = p_table->takeItem(row, col);
        p_table->setItem(row - 1, col, current);
        p_table->setItem(row, col, above);
    }
    p_table->selectRow(row - 1);
}

void HwArrayEditDialog::moveDown()
{
    int row = p_table->currentRow();
    if (row < 0 || row >= p_table->rowCount() - 1)
        return;

    for (int col = 0; col < p_table->columnCount(); ++col) {
        auto *below = p_table->takeItem(row + 1, col);
        auto *current = p_table->takeItem(row, col);
        p_table->setItem(row + 1, col, current);
        p_table->setItem(row, col, below);
    }
    p_table->selectRow(row + 1);
}

void HwArrayEditDialog::updateButtonStates()
{
    int row = p_table->currentRow();
    int total = p_table->rowCount();
    p_removeButton->setEnabled(row >= 0);
    p_upButton->setEnabled(row > 0);
    p_downButton->setEnabled(row >= 0 && row < total - 1);
}

void HwArrayEditDialog::accept()
{
    d_result.clear();
    for (int row = 0; row < p_table->rowCount(); ++row) {
        SettingsStorage::SettingsMap entry;
        for (int col = 0; col < d_subKeys.size(); ++col) {
            auto *item = p_table->item(row, col);
            entry[d_subKeys.at(col)] = item ? item->text() : QString();
        }
        d_result.push_back(entry);
    }
    QDialog::accept();
}
