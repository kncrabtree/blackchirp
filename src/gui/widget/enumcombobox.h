#ifndef ENUMCOMBOBOX_H
#define ENUMCOMBOBOX_H

#include <QComboBox>
#include <QMetaEnum>
#include <QTimerEvent>
#include <QStandardItem>
#include <QStandardItemModel>

/// \brief Type-safe combo box that auto-populates from a Q_ENUM-registered enumeration.
///
/// The constructor uses QMetaEnum::fromType<T>() to iterate over every enumerator in \c T,
/// adds one item per enumerator with its name as the display label (underscores replaced by
/// spaces) and its integer value stored as item data, and maintains type-safe accessors.
/// \c T must be declared with Q_ENUM or Q_ENUM_NS; otherwise QMetaEnum::fromType<T>()
/// returns an empty meta-object and the combobox contains no items.
template<typename T> class EnumComboBox : public QComboBox
{
public:
    /// \brief Construct the combo box and populate it from the enumerators of \c T.
    EnumComboBox(QWidget *parent = nullptr) : QComboBox(parent) {
        auto t = QMetaEnum::fromType<T>();
        auto num = t.keyCount();
        for(int i=0; i<num; ++i)
            addItem(QString(t.key(i)).replace(QChar('_'),QChar(' ')),
                    static_cast<T>(t.value(i)));

    }
    virtual ~EnumComboBox() {}

    /// \brief Return the enum value stored for item at index \a i.
    T value(int i) const { return itemData(i).template value<T>(); }
    /// \brief Return the enum value of the currently selected item.
    T currentValue() const { return currentData().template value<T>(); }
    /// \brief Select the item whose stored enum value equals \a v.
    ///
    /// Has no effect if \a v is not present in the combo box.
    void setCurrentValue(T v) {
        auto idx = findData(v);
        if(idx >= 0)
            setCurrentIndex(idx);
    }
    /// \brief Return the QStandardItem for the row whose enum value equals \a v, or nullptr.
    ///
    /// Callers can use the returned item to disable or restyle individual entries.
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
    /// \brief Return the QStandardItem at row index \a i, or nullptr if out of range.
    ///
    /// Callers can use the returned item to disable or restyle individual entries.
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
