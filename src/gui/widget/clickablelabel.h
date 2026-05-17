#ifndef CLICKABLELABEL_H
#define CLICKABLELABEL_H

#include <QLabel>

/*!
 * \brief A QLabel that acts as a link to a folder.
 *
 * With a non-empty folder path the rendered text behaves as a link:
 * the pointer becomes a hand and the text underlines on hover, and a
 * left click on the text opens the folder in the system file manager.
 * The interactive region is only the rendered text (honoring the
 * label's alignment), not the full widget width, so a stretched
 * centered label does not swallow unrelated clicks. An empty path
 * makes it an ordinary, non-interactive label, so one label can be
 * switched between the two states at runtime.
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
