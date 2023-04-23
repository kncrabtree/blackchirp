#ifndef ENUMCOMBOBOX_H
#define ENUMCOMBOBOX_H

#include <QComboBox>
#include <QMetaEnum>
#include <QTimerEvent>
#include <QStandardItem>
#include <QStandardItemModel>

template<typename T> class EnumComboBox : public QComboBox
{
public:
    EnumComboBox(QWidget *parent = nullptr) : QComboBox(parent) {
        auto t = QMetaEnum::fromType<T>();
        auto num = t.keyCount();
        for(int i=0; i<num; ++i)
            addItem(QString(t.key(i)).replace(QChar('_'),QChar(' ')),
                    static_cast<T>(t.value(i)));

    }
    virtual ~EnumComboBox() {}

    T value(int i) const { return itemData(i).template value<T>(); }
    T currentValue() const { return currentData().template value<T>(); }
    void setCurrentValue(T v) {
        auto idx = findData(v);
        if(idx >= 0)
            setCurrentIndex(idx);
    }    
    QStandardItem *itemForValue(T v) {
        auto row = findData(v);
        if(row >= 0)
        {
            auto m = dynamic_cast<QStandardItemModel*>(model());
            if(m)
                return dynamic_cast<QStandardItem*>(m->item(row));
        }
        return nullptr;
    }
    QStandardItem *itemAt(int i) {
        if(i >= 0)
        {
            auto m = dynamic_cast<QStandardItemModel*>(model());
            if(m)
                return dynamic_cast<QStandardItem*>(m->item(i));
        }
        return nullptr;
    }
};

#endif // ENUMCOMBOBOX_H
