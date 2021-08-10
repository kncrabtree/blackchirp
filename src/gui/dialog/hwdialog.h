#ifndef HWDIALOG_H
#define HWDIALOG_H

#include <QDialog>

class HWSettingsModel;

class HWDialog : public QDialog
{
    Q_OBJECT
public:
    HWDialog(QString key, QWidget *controlWidget = nullptr, QWidget *parent = nullptr);
    
private:
    HWSettingsModel *p_model;
       

    // QDialog interface
public slots:
    void accept() override;
    void reject() override;
    QSize sizeHint() const override;
};

#endif // HWDIALOG_H
