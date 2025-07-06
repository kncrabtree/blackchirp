#ifndef OVERLAYNUMericdelegate_H
#define OVERLAYNUMericdelegate_H

#include <QStyledItemDelegate>

class OverlayNumericDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit OverlayNumericDelegate(QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
};

#endif // OVERLAYNUMericdelegate_H