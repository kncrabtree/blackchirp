#ifndef TOOLBARWIDGETACTION_H
#define TOOLBARWIDGETACTION_H

#include <QWidgetAction>
#include <QMetaEnum>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QComboBox>

class QLabel;

class ToolBarWidgetAction : public QWidgetAction
{
    Q_OBJECT
public:
    ToolBarWidgetAction(const QString label, QWidget *parent = nullptr) : QWidgetAction(parent), d_label(label) {}
    QWidget *createWidget(QWidget *parent) override final;

protected:
    virtual QWidget *_createWidget(QWidget *parent) =0;
    QWidget *p_widget{nullptr};

private:
    QString d_label;
};

class SpinBoxWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    SpinBoxWidgetAction(const QString label = QString(""), QWidget *parent = nullptr)
        : ToolBarWidgetAction(label,parent) {}

    void setSpecialValueText(const QString text);
    void setRange(int min, int max);
    void setMinimum(int min);
    void setMaximum(int max);
    void setPrefix(const QString p);
    void setSuffix(const QString s);
    void setSingleStep(int step);
    int value() const;

protected:
    QWidget *_createWidget(QWidget *parent) override;


public slots:
    void setValue(int v);

signals:
    void valueChanged(int);

private:
    QString d_label, d_specialText, d_prefix, d_suffix;
    int d_value{0}, d_step{1};
    QPair<int,int> d_range{0,99};

};

class DoubleSpinBoxWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    DoubleSpinBoxWidgetAction(const QString label = QString(""), QWidget *parent = nullptr)
        : ToolBarWidgetAction(label,parent) {}

    void setSpecialValueText(QString text);;
    void setRange(double min, double max);;
    void setMinimum(double min);;
    void setMaximum(double max);;
    void setPrefix(const QString p);
    void setSuffix(const QString s);
    void setDecimals(int d);
    void setSingleStep(double s);
    double value() const;

protected:
    QWidget *_createWidget(QWidget *parent) override;


public slots:
    void setValue(double v) { d_value = v; emit valueChanged(d_value); }

signals:
    void valueChanged(double);

private:
    QString d_label, d_specialText, d_prefix, d_suffix;
    double d_value{0.0}, d_step{1.0};
    QPair<double,double> d_range{0,99};
    int d_decimals{2};

};

class ComboWABase : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    ComboWABase(const QString label, QWidget *parent) : ToolBarWidgetAction(label, parent) {
        p_model = new QStandardItemModel(this);
    }

    void setValue(const QVariant v) {
        for(auto w : createdWidgets())
        {
            auto box = w->findChild<QComboBox*>("ActionWidget");
            if(box)
                box->setCurrentIndex(box->findData(v));
        }
        d_currentValue = v;
        emit valueChanged(v);
    }

signals:
    void valueChanged(QVariant);

protected:
    QStandardItemModel *p_model{nullptr};
    QVariant d_currentValue;
    QString d_label;


protected:
    QWidget *_createWidget(QWidget *parent) override {
        auto out = new QComboBox(parent);
        out->setModel(p_model);
        out->setCurrentIndex(out->findData(d_currentValue));
        connect(out,qOverload<int>(&QComboBox::currentIndexChanged),[this,out](int i){ setValue(out->itemData(i)); });
        return out;
    }
};


template<typename T>
class EnumComboBoxWidgetAction : public ComboWABase
{
public:    
    EnumComboBoxWidgetAction(const QString label = QString(""), QWidget *parent = nullptr) :
        ComboWABase(label,parent)
    {
        auto me = QMetaEnum::fromType<T>();
        for(int i=0; i<me.keyCount(); ++i)
        {
            auto item = new QStandardItem(me.key(i));
            item->setData(static_cast<T>(me.value(i)),Qt::UserRole);
            p_model->appendRow(item);
            d_items.insert({static_cast<T>(me.value(i)),item});
        }
    }

    void setItemEnabled(T key, bool enabled = true) {
        auto it = d_items.find(key);
        if(it != d_items.end())
            it->second->setEnabled(enabled);
    }
    void setCurrentValue(T v) {
        setValue(QVariant::fromValue(v));
    }
    T value() const { return d_currentValue.value<T>(); }

private:
    std::map<T,QStandardItem*> d_items;

};

class CheckWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    CheckWidgetAction(const QString label, QWidget *parent = nullptr) :
        ToolBarWidgetAction(label,parent) { setCheckable(true); }

    QWidget *_createWidget(QWidget *parent) override;

};

class LabelWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    LabelWidgetAction(const QString label, QWidget *parent = nullptr) :
        ToolBarWidgetAction(label,parent) {}

    QWidget *_createWidget(QWidget *parent) override { return new QWidget(parent); }
};

#endif // TOOLBARWIDGETACTION_H
