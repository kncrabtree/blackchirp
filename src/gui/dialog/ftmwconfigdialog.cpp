#include "ftmwconfigdialog.h"

#include <QDialogButtonBox>
#include <QVBoxLayout>

#include <data/loadout/hardwareloadout.h>
#include <data/loadout/loadoutmanager.h>

#include <gui/widget/ftmwconfigwidget.h>

FtmwConfigDialog::FtmwConfigDialog(const QString &awgHwKey, const QString &digiHwKey,
                                   const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                                   QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("FTMW Configuration");
    resize(900, 650);

    auto *layout = new QVBoxLayout(this);

    p_widget = new FtmwConfigWidget(awgHwKey, digiHwKey, currentClocks, this);
    layout->addWidget(p_widget, 1);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttonBox);

    connect(p_widget, &FtmwConfigWidget::applyClocks, this, &FtmwConfigDialog::applyClocks);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &FtmwConfigDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void FtmwConfigDialog::accept()
{
    auto snap = p_widget->toSnapshot();
    auto active = LoadoutManager::instance().currentLoadout();
    if (active) {
        HardwareLoadout updated = *active;
        updated.ftmw = snap;
        LoadoutManager::instance().putLoadout(updated);
    }
    QDialog::accept();
}
