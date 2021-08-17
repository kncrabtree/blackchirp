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
    ToolBarWidgetAction(QWidget *parent = nullptr) : QWidgetAction(parent) {}
    QWidget *createWidget(QWidget *parent) override final;

protected:
    virtual QWidget *_createWidget(QWidget *parent) =0;
    virtual QString labelText() { return ""; }
};

class SpinBoxWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    SpinBoxWidgetAction(QString label = QString(""), QWidget *parent = nullptr)
        : ToolBarWidgetAction(parent), d_label(label) {}

    void setSpecialValueText(QString text) { d_specialText = text; };
    void setRange(int min, int max) { d_range = {min,max}; };
    void setMinimum(int min) { d_range.first = min; };
    void setMaximum(int max) { d_range.second = max; };
    void setPrefix(const QString p) { d_prefix = p; }
    void setSuffix(const QString s) { d_suffix = s; }
    int value() const { return d_value; }

protected:
    QString labelText() override { return d_label; }
    QWidget *_createWidget(QWidget *parent) override;


public slots:
    void setValue(int v) { d_value = v; emit valueChanged(d_value); }

signals:
    void valueChanged(int);

private:
    QString d_label, d_specialText, d_prefix, d_suffix;
    int d_value{0};
    QPair<int,int> d_range{0,99};

};

class DoubleSpinBoxWidgetAction : public ToolBarWidgetAction
{
    Q_OBJECT
public:
    DoubleSpinBoxWidgetAction(QString label = QString(""), QWidget *parent = nullptr)
        : ToolBarWidgetAction(parent), d_label(label) {}

    void setSpecialValueText(QString text) { d_specialText = text; };
    void setRange(double min, double max) { d_range = {min,max}; };
    void setMinimum(double min) { d_range.first = min; };
    void setMaximum(double max) { d_range.second = max; };
    void setPrefix(const QString p) { d_prefix = p; }
    void setSuffix(const QString s) { d_suffix = s; }
    void setDecimals(int d) { d_decimals = qBound(0,d,15); }
    void setSingleStep(double s) { d_step = s; }
    double value() const { return d_value; }

protected:
    QString labelText() override { return d_label; }
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
    ComboWABase(QString label, QWidget *parent) : ToolBarWidgetAction(parent), d_label(label) {
        p_model = new QStandardItemModel(this);
    }

    void setValue(const QVariant v) {
        if(p_box)
            p_box->setCurrentIndex(p_box->findData(v));
        d_currentValue = v;
        emit valueChanged(v);
    }
    QString labelText() override { return d_label; }

signals:
    void valueChanged(QVariant);

protected:
    QStandardItemModel *p_model{nullptr};
    QVariant d_currentValue;
    QString d_label;
    QComboBox *p_box{nullptr};


protected:
    QWidget *_createWidget(QWidget *parent) override {
        auto out = new QComboBox(parent);
        p_box = out;
        out->setModel(p_model);
        out->setCurrentIndex(out->findData(d_currentValue));
        connect(out,&QComboBox::destroyed,[this]{ p_box = nullptr;});
        connect(out,qOverload<int>(&QComboBox::currentIndexChanged),[this,out](int i){ setValue(out->itemData(i)); });
        return out;
    }
};


template<typename T>
class EnumComboBoxWidgetAction : public ComboWABase
{
public:
    EnumComboBoxWidgetAction(QString label = QString(""), QWidget *parent = nullptr) :
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

    void setEnabled(T key, bool enabled = true) {
        auto it = d_items.find(key);
        if(it != d_items.end())
            it->second->setEnabled(enabled);
    }
    void setCurrentValue(T v) {
        if(p_box)
            p_box->setCurrentIndex(p_box->findData(v));
        else
            setValue(v);
    }
    T value() const { return d_currentValue.value<T>(); }

private:
    std::map<T,QStandardItem*> d_items;


};

#endif // TOOLBARWIDGETACTION_H
