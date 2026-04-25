#ifndef FTMWCONFIGDIALOG_H
#define FTMWCONFIGDIALOG_H

#include <QDialog>
#include <QHash>

#include <data/experiment/rfconfig.h>

class FtmwConfigWidget;

namespace BC::Key::Ftmw
{
    using namespace Qt::StringLiterals;
    inline constexpr static QLatin1StringView ftmwDialogKey{"FtmwConfigDialog"};
}

class FtmwConfigDialog : public QDialog
{
    Q_OBJECT
public:
    explicit FtmwConfigDialog(const QString &awgHwKey, const QString &digiHwKey,
                               const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                               QWidget *parent = nullptr);

signals:
    void applyClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks);

private:
    void accept() override;

    FtmwConfigWidget *p_widget;
};

#endif // FTMWCONFIGDIALOG_H
