#ifndef EXPERIMENTCONFIGPAGE_H
#define EXPERIMENTCONFIGPAGE_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>

/// \brief Abstract base class for pages in the experiment-setup dialog.
///
/// Each concrete subclass represents one page of the experiment-setup wizard.
/// A page is identified by a persistent \c d_key (used to scope its
/// SettingsStorage) and a human-readable \c d_title. It holds a non-owning
/// pointer to the in-flight Experiment that is being configured.
///
/// The wizard drives pages through the lifecycle: initialize() → validate()
/// → apply(). Subclasses emit warning() or error() to surface diagnostic
/// messages in the wizard's status area without coupling the page to the
/// surrounding dialog.
class ExperimentConfigPage : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    /// \brief Construct the page.
    ///
    /// Subclasses should populate their widgets from the previous experiment or
    /// from SettingsStorage here. Only values that depend on choices made on
    /// other pages should be deferred to initialize().
    ///
    /// \param key    SettingsStorage scope key for this page.
    /// \param title  Human-readable page title shown in the wizard.
    /// \param exp    Non-owning pointer to the Experiment being configured.
    /// \param parent Parent widget.
    explicit ExperimentConfigPage(const QString key, const QString title, Experiment *exp, QWidget *parent = nullptr);

    const QString d_key;   ///< SettingsStorage scope key for this page.
    const QString d_title; ///< Human-readable title shown in the wizard header.

protected:
    Experiment *p_exp; ///< Non-owning pointer to the in-flight Experiment.

signals:
    /// \brief Emit a non-fatal warning message for the wizard to display.
    void warning(QString);
    /// \brief Emit a fatal error message for the wizard to display.
    void error(QString);

public slots:
    /// \brief Populate or refresh values that depend on choices made on other pages.
    ///
    /// The wizard calls this slot after all pages have been constructed and before
    /// the page is first shown. Values that can be determined from the previous
    /// experiment or from SettingsStorage should be set in the constructor instead.
    virtual void initialize() = 0;

    /// \brief Return true if the page's settings are internally consistent.
    ///
    /// The wizard calls this before advancing to the next page. Return false
    /// and emit error() or warning() to block the advance and display a
    /// diagnostic message.
    virtual bool validate() = 0;

    /// \brief Commit the page's settings into the Experiment.
    ///
    /// The wizard calls this after validate() returns true. Subclasses write
    /// their configuration into p_exp here.
    virtual void apply() = 0;

};

#endif // EXPERIMENTCONFIGPAGE_H
