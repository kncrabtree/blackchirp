#ifndef EXPTSUMMARYMODEL_H
#define EXPTSUMMARYMODEL_H

#include <QAbstractItemModel>
#include <memory>

#include <data/experiment/experiment.h>

class ExptTreeItem;

class ExptSummaryModel : public QAbstractItemModel
{
    Q_OBJECT
public:
    ExptSummaryModel(Experiment *exp, QObject *parent = nullptr);

private:
    std::unique_ptr<ExptTreeItem> pu_rootItem;

    // QAbstractItemModel interface
public:
    QModelIndex index(int row, int column, const QModelIndex &parent) const override;
    QModelIndex parent(const QModelIndex &child) const override;
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    ExptTreeItem *getItem(const QModelIndex &index) const;
};


class ExptTreeItem
{
public:
    ExptTreeItem(QStringList data, ExptTreeItem *parentItem = nullptr) :
        p_parent(parentItem), d_data(data) {}
    ~ExptTreeItem() { qDeleteAll(d_children); }

    void appendChild(ExptTreeItem *item) { d_children.append(item); }

    void setParent(ExptTreeItem *parent) { p_parent = parent; }
    ExptTreeItem *parent() const { return p_parent; }
    ExptTreeItem *childAt(int row) const { return d_children.value(row); }
    ExptTreeItem *findChild(const QString key) {
        for(auto child : d_children)
        {
            if(child->data(0) == key)
                return child;
        }
        return nullptr;
    }
    int childCount() const { return d_children.size(); }
    int columnCount() const { return d_data.size(); }
    QVariant data(int column) const { return d_data.value(column); }
    int row() const {
        return p_parent ? p_parent->d_children.indexOf(const_cast<ExptTreeItem*>(this)) : 0;
    }
    void sortChildren() {
        std::sort(d_children.begin(),d_children.end(),[](ExptTreeItem *a, ExptTreeItem *b){ return a->data(0) < b->data(0);});
    }

private:
    ExptTreeItem *p_parent;
    QStringList d_data;
    QVector<ExptTreeItem*> d_children;
};

#endif // EXPTSUMMARYMODEL_H
