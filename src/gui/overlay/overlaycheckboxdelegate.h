#ifndef OVERLAYCHECKBOXDELEGATE_H
#define OVERLAYCHECKBOXDELEGATE_H

#include <QStyledItemDelegate>

class OverlayCheckBoxDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit OverlayCheckBoxDelegate(QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    bool editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index) override;
};

#endif // OVERLAYCHECKBOXDELEGATE_H