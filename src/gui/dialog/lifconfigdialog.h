#ifndef LIFCONFIGDIALOG_H
#define LIFCONFIGDIALOG_H

#include <QDialog>

class LifConfigWidget;

namespace BC::Key::Lif
{
    using namespace Qt::StringLiterals;
    inline constexpr static QLatin1StringView lifDialogKey{"LifConfigDialog"};
}

class LifConfigDialog : public QDialog
{
    Q_OBJECT
public:
    explicit LifConfigDialog(const QString &digitizerHwKey, const QString &laserHwKey,
                              QWidget *parent = nullptr);

    LifConfigWidget *lifConfigWidget() const { return p_widget; }

private:
    void accept() override;

    LifConfigWidget *p_widget;
};

#endif // LIFCONFIGDIALOG_H
