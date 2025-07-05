#include "plotidcomboboxdelegate.h"

#include <QComboBox>

PlotIdComboBoxDelegate::PlotIdComboBoxDelegate(const QStringList &plotNames, QObject *parent)
    : QStyledItemDelegate(parent), d_plotNames(plotNames)
{
}

QWidget *PlotIdComboBoxDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(option)
    Q_UNUSED(index)
    
    QComboBox *comboBox = new QComboBox(parent);
    comboBox->addItems(d_plotNames);
    
    // Connect to automatically commit data when selection changes
    connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PlotIdComboBoxDelegate::commitAndCloseEditor);
    
    return comboBox;
}

void PlotIdComboBoxDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    QComboBox *comboBox = qobject_cast<QComboBox*>(editor);
    if (!comboBox)
        return;
    
    QString currentValue = index.model()->data(index, Qt::EditRole).toString();
    int currentIndex = comboBox->findText(currentValue);
    
    if (currentIndex >= 0) {
        comboBox->setCurrentIndex(currentIndex);
    }
}

void PlotIdComboBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    QComboBox *comboBox = qobject_cast<QComboBox*>(editor);
    if (!comboBox)
        return;
    
    QString selectedText = comboBox->currentText();
    model->setData(index, selectedText, Qt::EditRole);
}

void PlotIdComboBoxDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &rect) const
{
    Q_UNUSED(rect)
    editor->setGeometry(option.rect);
}

void PlotIdComboBoxDelegate::setPlotNames(const QStringList &plotNames)
{
    d_plotNames = plotNames;
}

void PlotIdComboBoxDelegate::commitAndCloseEditor()
{
    QComboBox *comboBox = qobject_cast<QComboBox*>(sender());
    if (comboBox) {
        emit commitData(comboBox);
        emit closeEditor(comboBox);
    }
}