#ifndef FTMWCONFIGDIALOG_H
#define FTMWCONFIGDIALOG_H

#include <QDialog>
#include <QHash>
#include <QString>

#include <data/experiment/rfconfig.h>

namespace Ui { class FtmwConfigDialog; }

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
    ~FtmwConfigDialog();

signals:
    void applyClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks);

private:
    void populateSourceCombos();
    void onRfSourceChanged(int index);
    void onChirpSourceChanged(int index);
    void onDigiSourceChanged(int index);
    void accept() override;

    Ui::FtmwConfigDialog *ui;
    QString d_awgHwKey;
    QString d_digiHwKey;
};

#endif // FTMWCONFIGDIALOG_H
