#ifndef HWSETTINGSMODEL_H
#define HWSETTINGSMODEL_H

#include <QAbstractItemModel>
#include <memory>
#include <data/storage/settingsstorage.h>

class HWSettingsItem;

class HWSettingsModel : public QAbstractItemModel, public SettingsStorage
{
    Q_OBJECT
public:
    HWSettingsModel(QString key, QStringList forbiddenKeys, QObject *parent = nullptr);

    void saveChanges();
    
private:
    std::unique_ptr<HWSettingsItem> pu_rootItem;
    
    // QAbstractItemModel interface
public:
    QModelIndex index(int row, int column, const QModelIndex &parent) const override;
    QModelIndex parent(const QModelIndex &child) const override;
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    bool hasChildren(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    bool insertRows(int row, int count, const QModelIndex &parent) override;
    bool removeRows(int row, int count, const QModelIndex &parent) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    HWSettingsItem *getItem(const QModelIndex &index) const;
};

class HWSettingsItem
{
public:
    HWSettingsItem(QVector<QVariant> data, bool editable, bool canAddChildren, HWSettingsItem *parent = nullptr) :
        p_parent(parent), d_data(data), d_isEditable(editable), d_canAddChildren(canAddChildren) {}
    ~HWSettingsItem() { qDeleteAll(d_children); }
    
    void appendChild(HWSettingsItem *item) { d_children.append(item); }
    
    HWSettingsItem *parent() const { return p_parent; }
    HWSettingsItem *childAt(int row) const { return d_children.value(row); }
    int childCount() const { return d_children.size(); }
    int columnCount() const { return d_data.size(); }
    QVariant data(int column) const { return d_data.value(column); }
    QStringList childKeys() const {
        QStringList out;
        for(auto child : d_children)
            out.append(child->data(0).toString());
        return out;
    };
    bool setData(int column, QVariant data) { 
        if(column < 0 || column >= d_data.size())
            return false;
         d_data[column] = data;
         return true;
    }
    int row() const {
        return p_parent ? p_parent->d_children.indexOf(const_cast<HWSettingsItem*>(this)) : 0;
    }
    bool isEditable() const { return d_isEditable; }
    bool canAddChildren() const { return d_canAddChildren; }
    bool addChild(int position, QStringList keys) {
        if(position < 0 || position > d_children.size())
            return false;

        QVector<QVariant> d{position,""};
        auto newItem = new HWSettingsItem(d,false,true,this);
        for(auto const &key : keys)
        {
            QVector<QVariant> data{key,""};
            newItem->appendChild(new HWSettingsItem(data,true,false,newItem));
        }
        d_children.insert(position,newItem);
        for(int i=0; i<d_children.size(); ++i)
            d_children[i]->setData(0,i);
        return true;
    }
    bool removeChild(int position) {
        if(position < 0 || position >= d_children.size())
            return false;

        delete d_children.takeAt(position);
        for(int i=0; i<d_children.size(); ++i)
            d_children[i]->setData(0,i);
        return true;
    }
      
private:
    HWSettingsItem *p_parent;
    QVector<HWSettingsItem*> d_children;
    QVector<QVariant> d_data;

    bool d_isEditable{false};
    bool d_canAddChildren{false};
    
};
    
    

#endif // HWSETTINGSMODEL_H
