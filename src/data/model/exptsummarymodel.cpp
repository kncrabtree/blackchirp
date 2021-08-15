#include "exptsummarymodel.h"

ExptSummaryModel::ExptSummaryModel(Experiment *exp, QObject *parent) : QAbstractItemModel(parent)
{
    if(!exp)
        return;

    QStringList l{"Object","Key","Value","Unit"};
    pu_rootItem = std::make_unique<ExptTreeItem>(l,nullptr);

    auto hdr = exp->getSummary();

    ExptTreeItem *currentObject = nullptr, *currentArray = nullptr, *currentArrayItem = nullptr;

    for(auto const &[obj,tpl] : hdr)
    {
        auto arrayKey = std::get<0>(tpl);
        auto arrayIdxStr = std::get<1>(tpl).isEmpty() ? QString("") : QString::number(std::get<1>(tpl).toInt() + 1);
        auto key = std::get<2>(tpl);
        auto val = std::get<3>(tpl);
        auto unit = std::get<4>(tpl);


        auto thisItem = arrayIdxStr.isEmpty() ? new ExptTreeItem({obj,key,val,unit}) :
                                                  new ExptTreeItem({"",key,val,unit});

        if(!currentObject || (currentObject->data(0).toString() != obj))
        {
            currentObject = new ExptTreeItem({obj,"","",""},pu_rootItem.get());
            pu_rootItem->appendChild(currentObject);
            currentArray = nullptr;
            currentArrayItem = nullptr;
        }

        if(arrayIdxStr.isEmpty())
        {
            thisItem->setParent(currentObject);
            currentObject->appendChild(thisItem);
        }
        else
        {
            if(!currentArray || (currentArray->data(0) != arrayKey))
            {
                currentArray = new ExptTreeItem({arrayKey,"","",""},currentObject);
                currentObject->appendChild(currentArray);
            }

            if(!currentArrayItem || (currentArrayItem->data(0) != arrayIdxStr))
            {
                currentArrayItem = new ExptTreeItem({arrayIdxStr,"","",""},currentArray);
                currentArray->appendChild(currentArrayItem);
            }

            thisItem->setParent(currentArrayItem);
            currentArrayItem->appendChild(thisItem);
        }

    }

}


QModelIndex ExptSummaryModel::index(int row, int column, const QModelIndex &parent) const
{
    if(!hasIndex(row,column,parent))
        return QModelIndex();

    ExptTreeItem *parentItem;
    if(parent.isValid())
        parentItem = static_cast<ExptTreeItem*>(parent.internalPointer());
    else
        parentItem = pu_rootItem.get();

    auto childItem = parentItem->childAt(row);

    return childItem ? createIndex(row,column,childItem) : QModelIndex();
}

QModelIndex ExptSummaryModel::parent(const QModelIndex &child) const
{
    if(!child.isValid())
        return QModelIndex();

    auto childItem = static_cast<ExptTreeItem*>(child.internalPointer());
    auto parentItem = childItem->parent();

    if(parentItem == pu_rootItem.get())
        return QModelIndex();

    return createIndex(parentItem->row(),0,parentItem);
}

int ExptSummaryModel::rowCount(const QModelIndex &parent) const
{
    return getItem(parent)->childCount();
}

int ExptSummaryModel::columnCount(const QModelIndex &parent) const
{
    return getItem(parent)->columnCount();
}

QVariant ExptSummaryModel::data(const QModelIndex &index, int role) const
{
    QVariant out;

    if(!index.isValid())
        return out;

    if(role == Qt::DisplayRole || role == Qt::EditRole)
        return getItem(index)->data(index.column());

    return out;
}

QVariant ExptSummaryModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return pu_rootItem->data(section);

    return QVariant();
}

ExptTreeItem *ExptSummaryModel::getItem(const QModelIndex &index) const
{
    if(index.isValid())
    {
        auto item = static_cast<ExptTreeItem*>(index.internalPointer());
        if(item)
            return item;
    }

    return pu_rootItem.get();
}
