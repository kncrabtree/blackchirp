#include "ftmwconfigwidget.h"

#include <set>

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QTabWidget>
#include <QVBoxLayout>

#include <data/bcglobals.h>
#include <data/experiment/ftmwconfig.h>
#include <gui/style/themecolors.h>
#include <data/loadout/loadoutmanager.h>
#include <data/loadout/rfconfigsnapshot.h>
#include <data/settings/hardwarekeys.h>

#include <gui/widget/chirpconfigwidget.h>
#include <gui/widget/ftmwdigitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>

using namespace Qt::StringLiterals;

FtmwConfigWidget::FtmwConfigWidget(const QString &awgHwKey, const QString &digiHwKey,
                                   const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                                   QWidget *parent)
    : QWidget(parent), SettingsStorage(BC::Key::FtmwConfigWidget::key),
      d_awgHwKey(awgHwKey), d_digiHwKey(digiHwKey)
{
    auto *mainLayout = new QVBoxLayout(this);

    p_tabWidget = new QTabWidget(this);

    // RF tab
    auto *rfTab = new QWidget;
    auto *rfTabLayout = new QVBoxLayout(rfTab);
    auto *rfSourceRow = new QHBoxLayout;
    rfSourceRow->addWidget(new QLabel("Load from loadout:"_L1, rfTab));
    p_rfSourceCombo = new QComboBox(rfTab);
    p_rfSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    rfSourceRow->addWidget(p_rfSourceCombo, 1);
    rfTabLayout->addLayout(rfSourceRow);
    p_rfWidget = new RfConfigWidget(rfTab);
    rfTabLayout->addWidget(p_rfWidget, 1);
    p_tabWidget->addTab(rfTab, "RF Config"_L1);

    // Chirp tab
    auto *chirpTab = new QWidget;
    auto *chirpTabLayout = new QVBoxLayout(chirpTab);
    auto *chirpSourceRow = new QHBoxLayout;
    chirpSourceRow->addWidget(new QLabel("Load from loadout:"_L1, chirpTab));
    p_chirpSourceCombo = new QComboBox(chirpTab);
    p_chirpSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    chirpSourceRow->addWidget(p_chirpSourceCombo, 1);
    chirpTabLayout->addLayout(chirpSourceRow);
    p_chirpWidget = new ChirpConfigWidget(chirpTab);
    chirpTabLayout->addWidget(p_chirpWidget, 1);
    p_tabWidget->addTab(chirpTab, "Chirp Config"_L1);

    // Digitizer tab
    auto *digiTab = new QWidget;
    auto *digiTabLayout = new QVBoxLayout(digiTab);
    auto *digiSourceRow = new QHBoxLayout;
    digiSourceRow->addWidget(new QLabel("Load from loadout:"_L1, digiTab));
    p_digiSourceCombo = new QComboBox(digiTab);
    p_digiSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    digiSourceRow->addWidget(p_digiSourceCombo, 1);
    digiTabLayout->addLayout(digiSourceRow);
    p_digiWidget = new FtmwDigitizerConfigWidget(digiTab);
    p_digiWidget->d_maxAnalogEnabled = 1;
    digiTabLayout->addWidget(p_digiWidget, 1);
    p_tabWidget->addTab(digiTab, "Digitizer Config"_L1);

    mainLayout->addWidget(p_tabWidget, 1);

    // Seed from loadout if the current loadout has changed since last use
    auto storedLoadout = get(BC::Key::FtmwConfigWidget::lastLoadout, QString());
    auto currentName = LoadoutManager::instance().currentLoadoutName();
    auto loadout = LoadoutManager::instance().currentLoadout();

    if (storedLoadout != currentName) {
        if (loadout && loadout->ftmw)
            initializeFromSnapshot(*loadout->ftmw);
        else {
            RfConfig rfc;
            rfc.setCurrentClocks(currentClocks);
            p_rfWidget->setClocks(currentClocks);
            p_chirpWidget->initialize(rfc);
        }
        set(BC::Key::FtmwConfigWidget::lastLoadout, currentName, true);
    }

    connect(p_rfWidget, &RfConfigWidget::applyClocks, this, &FtmwConfigWidget::applyClocks);

    populateSourceCombos();

    connect(p_rfSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onRfSourceChanged);
    connect(p_chirpSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onChirpSourceChanged);
    connect(p_digiSourceCombo, &QComboBox::currentIndexChanged,
            this, &FtmwConfigWidget::onDigiSourceChanged);

    connect(p_tabWidget, &QTabWidget::currentChanged, this, [this](int index) {
        if (index == 1) {
            RfConfig rfc;
            p_rfWidget->toRfConfig(rfc);
            p_chirpWidget->initialize(rfc);
            p_chirpWidget->updateChirpPlot();
        }
    });
}

void FtmwConfigWidget::initializeFromSnapshot(const FtmwSnapshot &snap)
{
    RfConfig rfc;
    snap.rfConfig.applyTo(rfc);
    p_rfWidget->setFromRfConfig(rfc);

    rfc.setChirpConfig(snap.chirpConfig);
    p_chirpWidget->setFromRfConfig(rfc);

    p_digiWidget->setFromConfig(snap.digitizer);
}

FtmwSnapshot FtmwConfigWidget::toSnapshot() const
{
    RfConfig rfc;
    p_rfWidget->toRfConfig(rfc);

    FtmwDigitizerConfig digiCfg(d_digiHwKey);
    p_digiWidget->toConfig(digiCfg);

    auto active = LoadoutManager::instance().currentLoadout();
    if (active && active->ftmw)
        digiCfg.d_fidChannel = active->ftmw->digitizer.d_fidChannel;

    FtmwSnapshot snap;
    snap.rfConfig = RfConfigSnapshot::fromRfConfig(rfc);
    snap.chirpConfig = p_chirpWidget->getChirps();
    snap.digitizer = digiCfg;
    snap.digiHwKey = d_digiHwKey;
    return snap;
}

void FtmwConfigWidget::initializeFromExperiment(const FtmwConfig &cfg)
{
    p_rfWidget->setFromRfConfig(cfg.d_rfConfig);
    p_chirpWidget->setFromRfConfig(cfg.d_rfConfig);
    p_digiWidget->setFromConfig(cfg.scopeConfig());
}

void FtmwConfigWidget::resetToLoadout()
{
    auto loadout = LoadoutManager::instance().currentLoadout();
    if (loadout && loadout->ftmw)
        initializeFromSnapshot(*loadout->ftmw);
}

void FtmwConfigWidget::updateChirpFromRf()
{
    RfConfig rfc;
    p_rfWidget->toRfConfig(rfc);
    p_chirpWidget->initialize(rfc);
    p_chirpWidget->updateChirpPlot();
}

void FtmwConfigWidget::setTabError(int tabIndex, bool hasError)
{
    if (hasError) {
        p_tabWidget->tabBar()->setTabTextColor(
            tabIndex, ThemeColors::getThemeAwareColor(ThemeColors::StatusError, this));
        p_tabWidget->setTabIcon(
            tabIndex, ThemeColors::createThemedIcon(
                ":/icons/exclamation-triangle.svg", ThemeColors::StatusError, this));
    } else {
        p_tabWidget->tabBar()->setTabTextColor(tabIndex, QColor());
        p_tabWidget->setTabIcon(tabIndex, QIcon());
    }
}

void FtmwConfigWidget::clearTabErrors()
{
    for (int i = 0; i < p_tabWidget->count(); ++i)
        setTabError(i, false);
}

void FtmwConfigWidget::populateSourceCombos()
{
    const QString sentinel = u"(no source — keep current)"_s;
    const QString activeName = LoadoutManager::instance().currentLoadoutName();

    auto awgMatches = LoadoutManager::instance().loadoutsMatchingHwKey(d_awgHwKey);
    awgMatches.removeAll(activeName);

    auto digiMatches = LoadoutManager::instance().loadoutsMatchingHwKey(d_digiHwKey);
    digiMatches.removeAll(activeName);

    for (auto *combo : {p_rfSourceCombo, p_chirpSourceCombo}) {
        combo->blockSignals(true);
        combo->clear();
        combo->addItem(sentinel);
        for (const auto &name : awgMatches)
            combo->addItem(name, name);
        combo->setCurrentIndex(0);
        combo->blockSignals(false);
    }

    p_digiSourceCombo->blockSignals(true);
    p_digiSourceCombo->clear();
    p_digiSourceCombo->addItem(sentinel);
    for (const auto &name : digiMatches)
        p_digiSourceCombo->addItem(name, name);
    p_digiSourceCombo->setCurrentIndex(0);
    p_digiSourceCombo->blockSignals(false);
}

void FtmwConfigWidget::onRfSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_rfSourceCombo->currentData().toString();
    p_rfSourceCombo->blockSignals(true);
    p_rfSourceCombo->setCurrentIndex(0);
    p_rfSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    const auto &sourceSnap = sourceLoadout->ftmw->rfConfig;

    RfConfig rfc;
    p_rfWidget->toRfConfig(rfc);
    auto dest = RfConfigSnapshot::fromRfConfig(rfc);

    std::set<QString> allowedHwKeys;
    auto active = LoadoutManager::instance().currentLoadout();
    if (active) {
        for (const auto &[k, v] : active->hardwareMap) {
            if (BC::Key::parseKey(k).first == BC::Key::Clock::clock)
                allowedHwKeys.insert(k);
        }
    }

    BC::Loadout::copyRfScalars(sourceSnap, dest);
    BC::Loadout::copyClocksMatching(sourceSnap, dest, allowedHwKeys);

    dest.applyTo(rfc);
    p_rfWidget->setFromRfConfig(rfc);
}

void FtmwConfigWidget::onChirpSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_chirpSourceCombo->currentData().toString();
    p_chirpSourceCombo->blockSignals(true);
    p_chirpSourceCombo->setCurrentIndex(0);
    p_chirpSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    RfConfig rfc;
    sourceLoadout->ftmw->rfConfig.applyTo(rfc);
    rfc.setChirpConfig(sourceLoadout->ftmw->chirpConfig);
    p_chirpWidget->setFromRfConfig(rfc);
}

void FtmwConfigWidget::onDigiSourceChanged(int index)
{
    if (index <= 0)
        return;

    const QString sourceName = p_digiSourceCombo->currentData().toString();
    p_digiSourceCombo->blockSignals(true);
    p_digiSourceCombo->setCurrentIndex(0);
    p_digiSourceCombo->blockSignals(false);

    auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout || !sourceLoadout->ftmw)
        return;

    p_digiWidget->setFromConfig(sourceLoadout->ftmw->digitizer);
}
