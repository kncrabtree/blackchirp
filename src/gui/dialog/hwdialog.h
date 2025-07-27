#ifndef HWDIALOG_H
#define HWDIALOG_H

#include <QDialog>

class HWSettingsModel;
class QTreeView;
class QLineEdit;
class QComboBox;
class QLabel;

class HWDialog : public QDialog
{
    Q_OBJECT
public:
    HWDialog(QString key, QStringList forbiddenKeys, QWidget *controlWidget = nullptr, QWidget *parent = nullptr);

    QString getHwName() const;
    int getSelectedProtocol() const;
    
private:
    QTreeView *p_view;
    HWSettingsModel *p_model;
    QLineEdit *p_nameEdit;
    QComboBox *p_protocolCombo;
    QLabel *p_protocolLabel;
    QString d_hwKey;
       

public slots:
    void insertBefore();
    void insertAfter();
    void remove();

    // QDialog interface
    void accept() override;
    void reject() override;
    QSize sizeHint() const override;
};

#endif // HWDIALOG_H
