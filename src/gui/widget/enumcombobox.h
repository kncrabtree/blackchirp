#ifndef ENUMCOMBOBOX_H
#define ENUMCOMBOBOX_H

#include <QComboBox>
#include <QMetaEnum>
#include <QTimerEvent>
#include <QStandardItem>
#include <QStandardItemModel>

/// \brief Non-template combo box populated from a runtime \c QMetaEnum.
///
/// Each row's display text is the enumerator key (underscores replaced by
/// spaces) and its item data is the raw key-name string, so any caller that
/// persists \c currentData() stores the \c Q_ENUM key name — the form read
/// back by \c BC::CSV::enumFromVariant. Use this directly where the enum
/// type is known only at runtime (e.g. reflective hardware-settings
/// rendering); the type-safe \c EnumComboBox<T> derives from it and reuses
/// the same population.
class EnumComboBoxBase : public QComboBox
{
public:
    explicit EnumComboBoxBase(QWidget *parent = nullptr) : QComboBox(parent) {}
    /// \brief Construct and populate from the enumerators of \a me.
    explicit EnumComboBoxBase(const QMetaEnum &me, QWidget *parent = nullptr) : QComboBox(parent) {
        populateFromMetaEnum(me);
    }
    virtual ~EnumComboBoxBase() {}

    /// \brief Select the row whose stored key name equals \a key. No effect if absent.
    void setCurrentKey(const QString &key) {
        auto idx = findData(key);
        if(idx >= 0)
            setCurrentIndex(idx);
    }

protected:
    /// \brief Replace the contents with one row per enumerator of \a me.
    void populateFromMetaEnum(const QMetaEnum &me) {
        clear();
        if(!me.isValid())
            return;
        for(int i=0; i<me.keyCount(); ++i) {
            const QString key = QString::fromUtf8(me.key(i));
            addItem(QString(key).replace(QChar('_'),QChar(' ')), key);
        }
    }
};

/// \brief Type-safe combo box that auto-populates from a Q_ENUM-registered enumeration.
///
/// Adds one row per enumerator of \c T (display label = key with underscores replaced by
/// spaces) and stores the key-name string as item data, giving type-safe accessors that
/// convert through \c QMetaEnum::fromType<T>(). \c T must be declared with Q_ENUM or
/// Q_ENUM_NS; otherwise the meta-enum is empty and the combo box contains no items.
template<typename T> class EnumComboBox : public EnumComboBoxBase
{
public:
    /// \brief Construct the combo box and populate it from the enumerators of \c T.
    EnumComboBox(QWidget *parent = nullptr) : EnumComboBoxBase(QMetaEnum::fromType<T>(), parent) {}
    virtual ~EnumComboBox() {}

    /// \brief Return the enum value stored for item at index \a i.
    T value(int i) const { return fromKey(itemData(i).toString()); }
    /// \brief Return the enum value of the currently selected item.
    T currentValue() const { return fromKey(currentData().toString()); }
    /// \brief Select the item whose stored enum value equals \a v.
    ///
    /// Has no effect if \a v is not present in the combo box.
    void setCurrentValue(T v) { setCurrentKey(toKey(v)); }
    /// \brief Return the QStandardItem for the row whose enum value equals \a v, or nullptr.
    ///
    /// Callers can use the returned item to disable or restyle individual entries.
    QStandardItem *itemForValue(T v) {
        auto row = findData(toKey(v));
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

private:
    static T fromKey(const QString &key) {
        return static_cast<T>(QMetaEnum::fromType<T>().keyToValue(key.toUtf8().constData()));
    }
    static QString toKey(T v) {
        return QString::fromUtf8(QMetaEnum::fromType<T>().valueToKey(static_cast<int>(v)));
    }
};

#endif // ENUMCOMBOBOX_H
