#include "ftmwconfigdialog.h"
#include "ftmwconfigdialog_ui.h"

#include <set>

#include <data/bcglobals.h>
#include <data/loadout/hardwareloadout.h>
#include <data/loadout/loadoutmanager.h>
#include <data/settings/hardwarekeys.h>

using namespace Qt::StringLiterals;

FtmwConfigDialog::FtmwConfigDialog(const QString &awgHwKey, const QString &digiHwKey,
                                   const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                                   QWidget *parent)
    : QDialog(parent), ui(new Ui::FtmwConfigDialog),
      d_awgHwKey(awgHwKey), d_digiHwKey(digiHwKey)
{
    ui->setupUi(this);

    auto loadout = LoadoutManager::instance().currentLoadout();
    if (loadout && loadout->ftmw)
    {
        const auto &snap = *loadout->ftmw;

        RfConfig rfc;
        snap.rfConfig.applyTo(rfc);
        ui->rfWidget->setFromRfConfig(rfc);

        rfc.setChirpConfig(snap.chirpConfig);
        ui->chirpWidget->setFromRfConfig(rfc);

        ui->digiWidget->setFromConfig(snap.digitizer);
    }
    else
    {
        RfConfig rfc;
        rfc.setCurrentClocks(currentClocks);
        ui->rfWidget->setClocks(currentClocks);
        ui->chirpWidget->initialize(rfc);
    }

    connect(ui->rfWidget, &RfConfigWidget::applyClocks, this, &FtmwConfigDialog::applyClocks);

    populateSourceCombos();

    connect(ui->rfSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigDialog::onRfSourceChanged);
    connect(ui->chirpSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigDialog::onChirpSourceChanged);
    connect(ui->digiSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigDialog::onDigiSourceChanged);

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, [this](int index) {
        if (index == 1) {
            RfConfig rfc;
            ui->rfWidget->toRfConfig(rfc);
            ui->chirpWidget->initialize(rfc);
        }
    });

    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &FtmwConfigDialog::accept);
    connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

FtmwConfigDialog::~FtmwConfigDialog()
{
    delete ui;
}

void FtmwConfigDialog::populateSourceCombos()
{
    const QString sentinel = u"(no source — keep current)"_s;
    const QString activeName = LoadoutManager::instance().currentLoadoutName();

    auto awgMatches = LoadoutManager::instance().loadoutsMatchingHwKey(d_awgHwKey);
    awgMatches.removeAll(activeName);

    auto digiMatches = LoadoutManager::instance().loadoutsMatchingHwKey(d_digiHwKey);
    digiMatches.removeAll(activeName);

    for (auto *combo : {ui->rfSourceCombo, ui->chirpSourceCombo})
    {
        combo->blockSignals(true);
        combo->clear();
        combo->addItem(sentinel);
        for (const auto &name : awgMatches)
            combo->addItem(name, name);
        combo->setCurrentIndex(0);
        combo->blockSignals(false);
    }

    ui->digiSourceCombo->blockSignals(true);
    ui->digiSourceCombo->clear();
    ui->digiSourceCombo->addItem(sentinel);
    for (const auto &name : digiMatches)
        ui->digiSourceCombo->addItem(name, name);
    ui->digiSourceCombo->setCurrentIndex(0);
    ui->digiSourceCombo->blockSignals(false);
}

void FtmwConfigDialog::onRfSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = ui->rfSourceCombo->currentData().toString();
    ui->rfSourceCombo->blockSignals(true);
    ui->rfSourceCombo->setCurrentIndex(0);
    ui->rfSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    const auto &sourceSnap = sourceLoadout->ftmw->rfConfig;

    RfConfig rfc;
    ui->rfWidget->toRfConfig(rfc);
    auto dest = RfConfigSnapshot::fromRfConfig(rfc);

    std::set<QString> allowedHwKeys;
    auto active = LoadoutManager::instance().currentLoadout();
    if (active)
    {
        for (const auto &[k, v] : active->hardwareMap)
        {
            if (BC::Key::parseKey(k).first == BC::Key::Clock::clock)
                allowedHwKeys.insert(k);
        }
    }

    BC::Loadout::copyRfScalars(sourceSnap, dest);
    BC::Loadout::copyClocksMatching(sourceSnap, dest, allowedHwKeys);

    dest.applyTo(rfc);
    ui->rfWidget->setFromRfConfig(rfc);
}

void FtmwConfigDialog::onChirpSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = ui->chirpSourceCombo->currentData().toString();
    ui->chirpSourceCombo->blockSignals(true);
    ui->chirpSourceCombo->setCurrentIndex(0);
    ui->chirpSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    RfConfig rfc;
    sourceLoadout->ftmw->rfConfig.applyTo(rfc);
    rfc.setChirpConfig(sourceLoadout->ftmw->chirpConfig);
    ui->chirpWidget->setFromRfConfig(rfc);
}

void FtmwConfigDialog::onDigiSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = ui->digiSourceCombo->currentData().toString();
    ui->digiSourceCombo->blockSignals(true);
    ui->digiSourceCombo->setCurrentIndex(0);
    ui->digiSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    ui->digiWidget->setFromConfig(sourceLoadout->ftmw->digitizer);
}

void FtmwConfigDialog::accept()
{
    RfConfig rfc;
    ui->rfWidget->toRfConfig(rfc);

    FtmwDigitizerConfig digiCfg(d_digiHwKey);
    ui->digiWidget->toConfig(digiCfg);

    // Preserve d_fidChannel from existing loadout snapshot if present
    auto active = LoadoutManager::instance().currentLoadout();
    if (active && active->ftmw)
        digiCfg.d_fidChannel = active->ftmw->digitizer.d_fidChannel;

    FtmwSnapshot snap;
    snap.rfConfig = RfConfigSnapshot::fromRfConfig(rfc);
    snap.chirpConfig = ui->chirpWidget->getChirps();
    snap.digitizer = digiCfg;
    snap.digiHwKey = d_digiHwKey;

    if (active)
    {
        HardwareLoadout updated = *active;
        updated.ftmw = snap;
        LoadoutManager::instance().putLoadout(updated);
    }

    QDialog::accept();
}
