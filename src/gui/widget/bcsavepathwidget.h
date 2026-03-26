#ifndef BCSAVEPATHWIDGET_H
#define BCSAVEPATHWIDGET_H

#include <QWidget>
#include <data/storage/settingsstorage.h>

class QSpinBox;
class QLineEdit;

class BCSavePathWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit BCSavePathWidget(QWidget *parent = nullptr);

    bool isReady() const;
    void save();

signals:
    void applied();

public slots:
    void apply();

private:
    QSpinBox *p_expBox;
    QLineEdit *p_lineEdit;
    bool d_applied{false};
};

#endif // BCSAVEPATHWIDGET_H
