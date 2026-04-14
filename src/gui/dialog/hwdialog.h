#ifndef HWDIALOG_H
#define HWDIALOG_H

#include <QDialog>

class HwSettingsWidget;

class HWDialog : public QDialog
{
    Q_OBJECT
public:
    HWDialog(QString key, QWidget *controlWidget = nullptr, QWidget *parent = nullptr);

    void discardControlWidget();

private:
    HwSettingsWidget *p_settingsWidget;
    QString d_hwKey;
    QWidget *p_controlWidget;

public slots:
    void accept() override;
    QSize sizeHint() const override;
};

#endif // HWDIALOG_H
