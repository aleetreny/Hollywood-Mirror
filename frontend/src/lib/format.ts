export function formatMovieTitle(title: string): string {
  return title.replace(/_\d+$/, '').replace(/_/g, ' ');
}
