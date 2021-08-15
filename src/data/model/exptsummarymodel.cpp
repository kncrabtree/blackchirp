#include "exptsummarymodel.h"

ExptSummaryModel::ExptSummaryModel(Experiment *exp, QObject *parent) : QAbstractItemModel(parent)
{
    if(!exp)
        return;

    QStringList l{"Object/Key","Value","Unit"};
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


        auto thisItem = new ExptTreeItem({key,val,unit});

        if(!currentObject || (currentObject->data(0).toString() != obj))
        {
            currentObject = new ExptTreeItem({obj,"",""},pu_rootItem.get());
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
                currentArray = new ExptTreeItem({arrayKey,"",""},currentObject);
                currentObject->appendChild(currentArray);
            }

            if(!currentArrayItem || (currentArrayItem->data(0) != arrayIdxStr))
            {
                currentArrayItem = new ExptTreeItem({arrayIdxStr,"",""},currentArray);
                currentArray->appendChild(currentArrayItem);
            }

            thisItem->setParent(currentArrayItem);
            currentArrayItem->appendChild(thisItem);
        }

    }

    if(exp->ftmwEnabled())
    {
        //chirps
        auto chirpConfigItem = pu_rootItem->findChild(BC::Store::CC::key);
        if(chirpConfigItem)
        {
            auto chirpRootItem = new ExptTreeItem({"Chirps","",""},chirpConfigItem);
            chirpConfigItem->appendChild(chirpRootItem);
            auto cl = exp->ftmwConfig()->d_rfConfig.d_chirpConfig.chirpList();
            for(int i=0; i<cl.size(); ++i)
            {
                ExptTreeItem *chirpItem = new ExptTreeItem({QString("Chirp %1").arg(i+1),"",""},chirpRootItem);
                chirpRootItem->appendChild(chirpItem);
                auto c = cl.at(i);
                for(int j=0; j<c.size(); ++j)
                {
                    ExptTreeItem *segItem = new ExptTreeItem({QString("Segment %1").arg(j+1),"",""},chirpItem);
                    chirpItem->appendChild(segItem);
                    auto &seg = c.at(j);
                    if(!seg.empty)
                    {
                        segItem->appendChild(new ExptTreeItem({"Start",QVariant(seg.startFreqMHz).toString(),"MHz"},segItem));
                        segItem->appendChild(new ExptTreeItem({"End",QVariant(seg.endFreqMHz).toString(),"MHz"},segItem));
                    }
                    else
                        segItem->appendChild(new ExptTreeItem({"Empty",QVariant(true).toString(),""},segItem));
                    segItem->appendChild(new ExptTreeItem({"Duration",QVariant(seg.durationUs).toString(),
                                                           QString::fromUtf8("μs")},segItem));

                }
            }

            chirpConfigItem->sortChildren();
        }

        //clock configuration
        auto rfConfigItem = pu_rootItem->findChild(BC::Store::RFC::key);
        if(rfConfigItem)
        {
            auto clockRootItem = new ExptTreeItem({"Clock Configs","",""},rfConfigItem);
            rfConfigItem->appendChild(clockRootItem);
            auto cs = exp->ftmwConfig()->d_rfConfig.clockSteps();
            for(int i=0; i<cs.size(); ++i)
            {
                auto clocks = cs.at(i);
                ExptTreeItem *stepItem = new ExptTreeItem({QString("Clock Step %1").arg(i+1),"",""},clockRootItem);
                clockRootItem->appendChild(stepItem);
                for(auto it = clocks.cbegin(); it != clocks.cend(); ++it)
                {
                    ExptTreeItem *clockItem = new ExptTreeItem({QVariant::fromValue<RfConfig::ClockType>(it.key()).toString()
                                                                ,"",""},stepItem);
                    stepItem->appendChild(clockItem);
                    auto &cf = it.value();
                    clockItem->appendChild(new ExptTreeItem({"Clock",cf.hwKey,""},clockItem));
                    clockItem->appendChild(new ExptTreeItem({"Output",QString::number(cf.output),""},clockItem));
                    clockItem->appendChild(new ExptTreeItem({"Frequency",QVariant(cf.desiredFreqMHz).toString(),"MHz"},clockItem));
                    clockItem->appendChild(new ExptTreeItem({"Operation",
                                                             QVariant::fromValue<RfConfig::MultOperation>(cf.op).toString(),
                                                             ""},clockItem));
                    clockItem->appendChild(new ExptTreeItem({"Factor",QVariant(cf.factor).toString(),""},clockItem));
                }
                stepItem->sortChildren();
            }
            rfConfigItem->sortChildren();
        }

    }


    for(int i=0;i<pu_rootItem->childCount(); ++i)
        pu_rootItem->childAt(i)->sortChildren();
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
