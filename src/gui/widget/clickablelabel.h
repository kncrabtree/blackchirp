#ifndef CLICKABLELABEL_H
#define CLICKABLELABEL_H

#include <QLabel>

/*!
 * \brief A QLabel that links to a folder.
 *
 * When given a non-empty folder path the label behaves as a link: over
 * the text the pointer becomes a hand and the text underlines, and a
 * left click there opens the folder in the system file manager
 * (mirroring the data-path label in the main window status bar). The
 * interactive region is just the rendered text, not the full widget
 * width, so a centered label stretched across a tab does not swallow
 * clicks aimed elsewhere. With an empty path it is an ordinary,
 * non-interactive label, so a caller can switch a single label between
 * the two states (e.g. a numbered experiment vs. Peak-Up mode, which
 * has no stored data).
 */
class ClickableLabel : public QLabel
{
    Q_OBJECT
public:
    explicit ClickableLabel(QWidget *parent = nullptr);

    /*!
     * \brief Set the folder opened on click.
     *
     * An empty path disables interactivity and clears the tooltip.
     */
    void setFolderPath(const QString &path);

protected:
    void enterEvent(QEnterEvent *e) override;
    void leaveEvent(QEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void mouseReleaseEvent(QMouseEvent *e) override;

private:
    QString d_folderPath;
    bool d_hot{false};

    /// Bounding rect of the rendered text within the widget, honoring
    /// the label's alignment. The hit/hover target, computed from a
    /// non-underlined font so it does not shift on hover.
    QRect textRect() const;
    void updateHover(const QPoint &pos);
    void setUnderline(bool on);
};

#endif // CLICKABLELABEL_H
