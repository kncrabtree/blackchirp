#ifndef PLOTIDCOMBOBOXDELEGATE_H
#define PLOTIDCOMBOBOXDELEGATE_H

#include <QStyledItemDelegate>
#include <QComboBox>
#include <QStringList>

class PlotIdComboBoxDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit PlotIdComboBoxDelegate(const QStringList &plotNames, QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &rect) const override;

    // Update plot names if they change
    void setPlotNames(const QStringList &plotNames);

private slots:
    void commitAndCloseEditor();

private:
    QStringList d_plotNames;
};

#endif // PLOTIDCOMBOBOXDELEGATE_H