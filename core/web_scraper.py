# #### core/web_scraper.py

## uncomment the code below if using the links, and replace the link with suitable links

# import requests
# from bs4 import BeautifulSoup
# import time
# from typing import List, Dict, Any, Optional, Tuple, Set
# import logging
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from urllib.parse import urljoin, urlparse
# import hashlib


# from utils.config import config
# from utils.helpers import generate_document_id

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class IndigoWebScraper:
#     """Scrapes content from the Indigo Airlines website for specific sections."""
    
#     def __init__(self):
#         self.base_url = "https://www.goindigo.in"
#         self.session = requests.Session()
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#             "Accept-Language": "en-US,en;q=0.9",
#         }
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=config.CHUNK_SIZE,
#             chunk_overlap=config.CHUNK_OVERLAP,
#             length_function=len,
#         )

#         self.target_sections = {
#             "offers": "https://www.goindigo.in/campaigns/indigo-offers.html",
#             "6e_offers": "https://www.goindigo.in/add-on-services/6exclusive-fare.html?linkNav=6exclusive+fare%7C1%7CFlight+offers",
#             "student_discount": "https://www.goindigo.in/add-on-services/student-discount.html",
#             "armed_forces": "https://www.goindigo.in/add-on-services/armed-forces.html",
#             "ajio_luxe": "https://www.goindigo.in/campaigns/ajioluxe-offers.html",
#             "excess_baggage": "https://www.goindigo.in/add-on-services/excess-baggage.html?linkNav=Excess+baggage%7C1%7CAddons",
#             "partner_offers": "https://www.goindigo.in/hotels.html?linkNav=6e-bagport%7C1%7CPartner+Offers",
#             "senior_citizen": "https://www.goindigo.in/add-on-services/senior-citizen-discount.html",
#             "fast_forward": "https://www.goindigo.in/add-on-services/fast-forward.html?linkNav=Fast+forward%7C2%7CAddons",
#             "offers_indigo_spotify": "https://www.goindigo.in/indigo-spotify/terms-and-conditions.html",
#             "skyplus_6e": "https://www.goindigo.in/content/skyplus6e/in/en/mokobara.html?linkNav=6e-bagport%7C1%7CPartner+Offers",
#             "offers_short_stay_offer": "https://www.goindigo.in/hotels?linkNav=IndiGomainofferpage%7CGOSTAY",
#             "terms_conditions": "https://www.goindigo.in/information/terms-and-conditions.html",
#             "internation_travel_tips": "https://www.goindigo.in/information/useful-tips-for-your-international-flight.html",
#             "web_checkin_advisory": "https://www.goindigo.in/web-check-in.html",
#             "conditions_of_carriage": "https://www.goindigo.in/information/conditions-of-carriage.html",
#             "faqs": "https://www.goindigo.in/travel-information/en.html",
#             "faqs_indigo_stretch": "https://www.goindigo.in/indigo-stretch.html",
#             "faqs_check_in_options": "https://www.goindigo.in/travel-information/en/check-in-options.html",
#             "faqs_flight_status": "https://www.goindigo.in/check-flight-status.html",
#             "faqs_fare_rules": "https://www.goindigo.in/travel-information/en/fare-rules.html",
#             "faqs_flight_delays_and_cancellation": "https://www.goindigo.in/travel-information/en/flight-delays-cancellations.html",
#             "faqs_refunds": "https://www.goindigo.in/travel-information/en/refunds.html",
#             "faqs_ticket_modification": "https://www.goindigo.in/travel-information/en/ticket-modifications.html",
#             "faqs_pre_paid_baggage": "https://www.goindigo.in/travel-information/en/pre-paid-baggage.html",
#             "faqs_infants_travel": "https://www.goindigo.in/travel-information/en/infants.html",
#             "faqs_travel_certificate": "https://www.goindigo.in/travel-information/en/travel-certificate.html",
#             "faqs_add_on_services": "https://www.goindigo.in/travel-information/en/add-on-services.html",
#             "faqs_on_board_facilities": "https://www.goindigo.in/travel-information/en/on-board-facilities.html",
#             "faqs_airport_check_in_requirements": "https://www.goindigo.in/travel-information/en/airport-check-in-requirements.html",
#             "faqs_terminal_infor": "https://www.goindigo.in/travel-information/en/terminal-information.html",
#             "faqs_non_standard_and_specail_baggage": "https://www.goindigo.in/travel-information/en/non-standard-baggage.html",
#             "faqs_lost_and_mishandled_baggage": "https://www.goindigo.in/travel-information/en/lost-mishandled-baggage.html",
#             "faqs_aircraft_and_routes": "https://www.goindigo.in/travel-information/en/aircraft-and-routes.html",
#             "faqs_contact_us": "https://www.goindigo.in/travel-information/en/contact-us.html",
#             "faqs_fare_rules": "https://www.goindigo.in/travel-information/en/fare-rules.html",
#             "faqs_expectant_mother": "https://www.goindigo.in/travel-information/en/expectant-mother.html",
#             "faqs_medical_assistance": "https://www.goindigo.in/travel-information/en/passengers-with-medical-conditions.html",
#             "faqs_passenger_with_special_needs": "https://www.goindigo.in/travel-information/en/passengers-with-special-needs.html",
#             "faqs_unaccompanied_minor_travel": "https://www.goindigo.in/travel-information/en/unaccompanied-minor.html",
#             "faqs_payment_alerts": "https://www.goindigo.in/travel-information/en/payment-alerts.html",
#             "faqs_reservations": "https://www.goindigo.in/travel-information/en/reservations.html",
#             "faqs_payment_support": "https://www.goindigo.in/travel-information/en/payments.html",
#             "faqs_travel_documents": "https://www.goindigo.in/travel-information/en/travel-documents.html",
#             "faqs_6e_fares": "https://www.goindigo.in/travel-information/en/6e-fares.html",
#             "faqs_codeshare_flight": "https://www.goindigo.in/travel-information/en/codeshare.html",
#             "faqs_dubai_airport_change": "https://www.goindigo.in/travel-information/en/dubai-airport-change.html",
#             "faqs_6e_rewards": "https://www.goindigo.in/travel-information/en/6e-rewards.html",
#             "faqs_credit_shell": "https://www.goindigo.in/travel-information/en/credit-shell.html",
#             "faqs_iata_travel_pass": "https://www.goindigo.in/travel-information/en/iata-travel-pass.html",
#             "faqs_fees_and_charges": "https://www.goindigo.in/travel-information/en/fees-and-charges.html",
#             "faqs_hostels": "https://www.goindigo.in/travel-information/en/hotels.html",
#             "faqs_indigo_stretch": "https://www.goindigo.in/travel-information/en/indigo-stretch.html",
#             "indigo_cargo": "https://cargo.goindigo.in/",
#             "book_cargo": "https://cargo.goindigo.in/content/indigo/cargo/en/our-locations/domestic-contacts.html",
#             "general_cargo": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/general-cargo.html",
#             "perishable_cargo": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/perishable.html",
#             "cargo_dangerous": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods.html",
#             "cargo_dangerous_dryice": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/dry-ice.html",
#             "cargo_dangerous_flammable_liquid": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/flammable-liquid.html",
#             "cargo_dangerous_class7_radioactive_materials": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/radioactive-cargo.html",
#             "cargo_dangerous_lithium_batteries": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/lithium-batteries.html",
#             "cargo_vulnerable": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/vulnerable-cargo.html",
#             "cargo_FAC": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/fac.html",
#             "cargo_6e_priority": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/6e-priority.html",
#             "cargo_human_rights": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/human-remains.html",
#             "cargo_cremated_ashes": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/cremated-ashes.html",
#             "cargo_valuable": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/valuable-cargo.html",
#             "cargo_dry_shippers": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/dry-shippers.html",
#             "cargo_dry_batteries": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/special-products/dry-batteries.html",
#             "cargo_class3_flammable_liquid": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/flammable-liquid.html",
#             "cargo_biological_products": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/biological-products.html",
#             "cargo_environmentally_hazardous": "https://cargo.goindigo.in/content/indigo/cargo/en/cargo/dangerous-goods/environmentally-hazardous-substances.html",
#             "contact_us": "https://www.goindigo.in/contact-us.html",
#             "6e_skai": "https://www.goindigo.in/support.html",
#             "hotels": "https://www.goindigo.in/hotels",
#             "charters_services": "https://www.goindigo.in/charters.html",
#             "refund_claim": "https://www.goindigo.in/initiate-refund.html",
#             "baggage": "https://www.goindigo.in/baggage.html",
#             "baggage_allowance": "https://www.goindigo.in/baggage/baggage-allowance.html",
#             "baggage_excess": "https://www.goindigo.in/baggage/excess-baggage.html",
#             "baggage_cabin": "https://www.goindigo.in/baggage/cabin-baggage.html",
#             # "baggage_track": "https://wtrweb.worldtracer.aero/WTRInternet/wtwflowinternet.do?_flowExecutionKey=_c2789311D-DDB4-10C5-9F26-F6C9BF3A496C_k50431363-5451-442E-9F60-1E1B924B95A7",
#             "baggage_lost": "https://www.goindigo.in/baggage/delay-lost-baggage.html",
#             "baggage_non_standard": "https://www.goindigo.in/baggage/special-baggage.html",
#             "baggage_things_not_allowed": "https://www.goindigo.in/baggage/dangerous-goods-policy.html",
#             "baggage_transfer": "https://www.goindigo.in/baggage/baggage-transfer.html",
#             "baggage_protection": "https://www.goindigo.in/add-on-services/delayed-and-lost-baggage-protection.html",
#             "6e_fare_indigo_stretch": "https://www.goindigo.in/indigo-stretch/terms-and-conditions.html",
#             "6e_fare_super_fare": "https://www.goindigo.in/add-on-services/super-6e-fare.html",
#             "6e_fare_flexi_fares": "https://www.goindigo.in/add-on-services/flexi-fares.html",
#             "6e_fare_infant": "https://www.goindigo.in/add-on-services/infants.html",
#             "6e_xtras_indigo_early": "https://www.goindigo.in/add-on-services/indigo-early.html",
#             "6e_xtras_bag_port": "https://www.goindigo.in/campaigns/indigo-offers/6e-bagport.html",
#             "add_on_services": "https://www.goindigo.in/add-on-services.html",
#             "6e_prime": "https://www.goindigo.in/add-on-services/6e-prime.html",
#             "6e_cancellation_assist": "https://www.goindigo.in/add-on-services/free-cancellation-insurance.html",
#             "6e_eat": "https://www.goindigo.in/add-on-services/food-menu.html",
#             "6e_blanket_eye_shade": "https://www.goindigo.in/add-on-services/blanket-pillow-eye-shade.html",
#             "6e_one_for_skies": "https://www.goindigo.in/add-on-services/6e-bar.html",
#             "6e_sports_equip": "https://www.goindigo.in/add-on-services/sports-equipment-handling-fees.html",
#             "6e_travel_assistance": "https://www.goindigo.in/add-on-services/travel-assistance.html",
#             "6e_seat_and_eat": "https://www.goindigo.in/add-on-services/6e-seat-and-eat.html",
#             "6e_delayed_and_lost_baggage": "https://www.goindigo.in/add-on-services/delayed-and-lost-baggage-protection.html",
#             "seat_select": "https://www.goindigo.in/add-on-services/seat-plus.html",
#             "atr_seat_layout": "https://www.goindigo.in/aircraft-and-fleet.html",
#             "double_seat": "https://www.goindigo.in/information/6e-multi-seats.html",
#             # "medical_assistance": "https://www.goindigo.in/content/dam/s6web/in/en/assets/documents/MedicalPassenger-TravelGuideline.pdf?linkNav=Medical%20Assistance%7CSUPPORT%7CFooter",
#             "special_disability_assistance": "https://www.goindigo.in/information/special-disability-assistance.html",
#             "disability_assistance": "https://www.goindigo.in/information/special-disability-assistance/disability-assistance.html",
#             "special_assistance": "https://www.goindigo.in/information/special-disability-assistance/special-assistance.html",
#             "plan_b": "https://www.goindigo.in/plan-b.html"
#         }
    
#     def _get_page_content(self, url: str) -> Optional[str]:
#         """Fetch content from a URL with error handling and rate limiting."""
#         try:
#             full_url = url if url.startswith(('http://', 'https://')) else urljoin(self.base_url, url)
#             logger.info(f"Fetching content from: {full_url}")
            
#             # Rate limiting to be respectful to the website
#             time.sleep(1)
            
#             response = self.session.get(full_url, headers=self.headers, timeout=10)
#             response.raise_for_status()
            
#             return response.text
#         except requests.RequestException as e:
#             logger.error(f"Error fetching {url}: {str(e)}")
#             return None

#     def _extract_content(self, html: str, section_name: str, url: str) -> Dict[str, Any]:
#         """Extract relevant content from HTML based on section type."""
#         soup = BeautifulSoup(html, 'html.parser')
        
#         # Find the main content area
#         content_area = soup.select_one('.content-area, .page-content, article, .main-content')
        
#         if not content_area:
#             logger.warning(f"Could not find main content area for {section_name}")
#             content_area = soup.body
        
#         # Extract the text content and HTML for change detection
#         content_text = content_area.get_text(separator='\n', strip=True)
#         content_html = str(content_area)
        
#         # Create metadata
#         metadata = {
#             "source": f"indigo-website-{section_name}",
#             "url": url,
#             "section": section_name,
#             "scrape_timestamp": time.time(),
#             "content_hash": hashlib.md5(content_html.encode()).hexdigest()  # For change detection
#         }
        
#         return {"text": content_text, "metadata": metadata}
    
#     def _process_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Split content into chunks and prepare for vectorization."""
#         if not content or not content.get("text"):
#             return []
        
#         chunks = self.text_splitter.split_text(content["text"])
        
#         processed_chunks = []
#         for i, chunk in enumerate(chunks):
#             chunk_id = generate_document_id(f"{content['metadata']['section']}-{i}-{chunk}")
            
#             processed_chunks.append({
#                 "text": chunk,
#                 "metadata": {
#                     **content["metadata"],
#                     "chunk_id": chunk_id,
#                     "chunk_index": i,
#                     "total_chunks": len(chunks),
#                     "parent_hash": content["metadata"]["content_hash"]  # Track which page this came from
#                 }
#             })
        
#         return processed_chunks
    
#     def scrape_with_changes(self, existing_hashes: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
#         """
#         Scrape all sections and return:
#         - List of new/updated chunks
#         - List of parent hashes that no longer exist (indicating deleted content)
#         """
#         all_chunks = []
#         current_hashes = {}
#         deleted_hashes = []
        
#         for section_name, url_path in self.target_sections.items():
#             html_content = self._get_page_content(url_path)
            
#             if html_content:
#                 content = self._extract_content(html_content, section_name, url_path)
#                 current_hash = content["metadata"]["content_hash"]
#                 current_hashes[section_name] = current_hash
                
#                 # Only process if content has changed or is new
#                 if section_name not in existing_hashes or existing_hashes[section_name] != current_hash:
#                     chunks = self._process_content(content)
#                     all_chunks.extend(chunks)
#                     logger.info(f"Processed {len(chunks)} chunks from {section_name} (changed)")
#                 else:
#                     logger.info(f"Skipping unchanged section: {section_name}")
#             else:
#                 logger.warning(f"Failed to fetch content for {section_name}")
        
#         # Detect deleted sections (present in existing_hashes but not in current scrape)
#         for section_name in existing_hashes:
#             if section_name not in current_hashes:
#                 deleted_hashes.append(existing_hashes[section_name])
#                 logger.info(f"Detected deleted section: {section_name}")
        
#         return all_chunks, deleted_hashes
    
#     def scrape_all_sections(self) -> List[Dict[str, Any]]:
#         """Scrape all target sections and return processed chunks."""
#         all_chunks = []
        
#         for section_name, url_path in self.target_sections.items():
#             html_content = self._get_page_content(url_path)
            
#             if html_content:
#                 content = self._extract_content(html_content, section_name, url_path)
#                 chunks = self._process_content(content)
#                 all_chunks.extend(chunks)
                
#                 logger.info(f"Processed {len(chunks)} chunks from {section_name}")
#             else:
#                 logger.warning(f"Failed to fetch content for {section_name}")
        
#         return all_chunks
    
#     def scrape_section(self, section_name: str) -> List[Dict[str, Any]]:
#         """Scrape a specific section by name."""
#         if section_name not in self.target_sections:
#             logger.error(f"Unknown section: {section_name}")
#             return []
        
#         url_path = self.target_sections[section_name]
#         html_content = self._get_page_content(url_path)
        
#         if not html_content:
#             return []
        
#         content = self._extract_content(html_content, section_name, url_path)
#         chunks = self._process_content(content)
        
#         return chunks
        
#     def _find_and_follow_links(self, start_url: str, max_depth: int = 1, max_links: int = 10) -> List[str]:
#         """Find and follow links within the IndiGo website up to a certain depth."""
#         visited = set()
#         to_visit = [(start_url, 0)]  # (url, depth)
#         discovered_urls = []
        
#         while to_visit and len(discovered_urls) < max_links:
#             current_url, depth = to_visit.pop(0)
            
#             # Skip if already visited or max depth reached
#             if current_url in visited or depth > max_depth:
#                 continue
                
#             visited.add(current_url)
            
#             # Get page content
#             html_content = self._get_page_content(current_url)
#             if not html_content:
#                 continue
                
#             # Add to discovered URLs
#             discovered_urls.append(current_url)
            
#             # If at max depth, don't look for more links
#             if depth == max_depth:
#                 continue
                
#             # Parse links
#             soup = BeautifulSoup(html_content, 'html.parser')
#             for link in soup.find_all('a', href=True):
#                 href = link['href']
                
#                 # Skip empty links, anchors, and external links
#                 if not href or href.startswith('#'):
#                     continue
                    
#                 # Convert to absolute URL
#                 if not href.startswith(('http://', 'https://')):
#                     href = urljoin(current_url, href)
                
#                 # Skip external links