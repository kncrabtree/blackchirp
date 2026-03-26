#ifndef BCSAVEPATHDIALOG_H
#define BCSAVEPATHDIALOG_H

#include <QDialog>

class BCSavePathWidget;
class QDialogButtonBox;

class BCSavePathDialog : public QDialog
{
public:
    BCSavePathDialog(QWidget *parent = nullptr);
    QSize sizeHint() const override;

public slots:
    void accept() override;
    void reject() override;

private:
    BCSavePathWidget *p_widget;
    QDialogButtonBox *p_buttons;
};

#endif // BCSAVEPATHDIALOG_H
